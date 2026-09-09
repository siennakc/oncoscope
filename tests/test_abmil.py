"""ABMIL: learns bag labels AND points its attention at the right tiles.

Runs on any machine with torch (CI without torch skips). Synthetic bags make
the ground truth exact: positive bags contain a few 'tumor' tiles drawn from
a shifted distribution; every other tile is background noise. A correct ABMIL
must (1) classify held-out bags well and (2) concentrate attention mass on
the planted tiles — the interpretability property is tested, not assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oncoscope.models.abmil import (  # noqa: E402
    GatedABMIL, attention, predict, train_abmil,
)

DIM = 32

# The 'tumor' signature is a fixed DIRECTION in embedding space, not a mean
# shift: the model's LayerNorm removes per-tile means (as it should — FM
# embeddings differ in scale, not offset), so a flat shift would be erased by
# the model's own preprocessing and test nothing.
_SIGNAL_DIR = np.random.default_rng(99).normal(0, 1, DIM)
_SIGNAL_DIR = (_SIGNAL_DIR - _SIGNAL_DIR.mean()) / np.linalg.norm(_SIGNAL_DIR)


def _bag(rng, positive: bool, n=60, n_signal=4):
    bag = rng.normal(0, 1, (n, DIM)).astype(np.float32)
    signal_idx = np.array([], dtype=int)
    if positive:
        signal_idx = rng.choice(n, size=n_signal, replace=False)
        bag[signal_idx] += (8.0 * _SIGNAL_DIR).astype(np.float32)
    return bag, signal_idx


def _world(seed=0, n_train=40, n_val=20):
    rng = np.random.default_rng(seed)
    def make(k):
        bags, labels, signals = [], [], []
        for i in range(k):
            positive = i % 2 == 0
            bag, sig = _bag(rng, positive)
            bags.append(bag); labels.append(int(positive)); signals.append(sig)
        return bags, labels, signals
    return make(n_train), make(n_val)


@pytest.fixture(scope="module")
def trained():
    (tr_bags, tr_labels, _), (va_bags, va_labels, va_signals) = _world()
    model, history = train_abmil(
        tr_bags, tr_labels, embed_dim=DIM, epochs=25, device="cpu",
        val_bags=va_bags, val_labels=va_labels)
    return model, history, va_bags, va_labels, va_signals


def test_learns_to_classify_heldout_bags(trained):
    model, history, va_bags, va_labels, _ = trained
    assert history[-1]["val_auroc"] > 0.9, f"history tail: {history[-3:]}"


def test_attention_concentrates_on_signal_tiles(trained):
    model, _, va_bags, va_labels, va_signals = trained
    hits = total = 0
    for bag, label, sig in zip(va_bags, va_labels, va_signals):
        if not label:
            continue
        attn = attention(model, bag, device="cpu")
        # the planted tiles are 4 of 60 (~7%); a correct model routes far more
        # attention mass than that through them
        hits += float(attn[sig].sum())
        total += 1
    mean_mass = hits / total
    assert mean_mass > 0.5, f"attention mass on planted tiles only {mean_mass:.2f}"


def test_attention_sums_to_one_and_matches_bag(trained):
    model, _, va_bags, _, _ = trained
    attn = attention(model, va_bags[0], device="cpu")
    assert attn.shape == (va_bags[0].shape[0],)
    assert abs(attn.sum() - 1.0) < 1e-4
    assert (attn >= 0).all()


def test_prediction_is_deterministic(trained):
    model, _, va_bags, _, _ = trained
    p1 = predict(model, va_bags[:5], device="cpu")
    p2 = predict(model, va_bags[:5], device="cpu")
    assert np.allclose(p1, p2)


def test_auroc_uses_midranks_for_ties():
    """Saturated sigmoids tie; a tie broken by array order is not a metric."""
    from oncoscope.eval.metrics import auroc
    from oncoscope.models.abmil import _auroc
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    s = np.array([1.0, 1.0, 1.0, 0.2, 0.5, 0.5, 0.9, 0.9])
    assert abs(_auroc(y, s) - auroc(y, s)) < 1e-12
    perm = np.random.default_rng(0).permutation(len(y))
    assert abs(_auroc(y[perm], s[perm]) - _auroc(y, s)) < 1e-12
    assert _auroc(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5


def test_ragged_bags_and_subsampling():
    """Bags of wildly different sizes train without shape errors, and bags
    over max_tiles are handled (the subsample path)."""
    rng = np.random.default_rng(3)
    bags, labels = [], []
    for i, n in enumerate([5, 300, 47, 1200, 80, 640]):
        bag, _ = _bag(rng, positive=i % 2 == 0, n=n)
        bags.append(bag); labels.append(int(i % 2 == 0))
    model, history = train_abmil(bags, labels, embed_dim=DIM, epochs=2,
                                 max_tiles=256, device="cpu")
    probs = predict(model, bags, device="cpu")
    assert probs.shape == (6,)
    assert np.isfinite(probs).all()
