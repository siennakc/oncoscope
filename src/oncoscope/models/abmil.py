"""Gated attention-based multiple-instance learning (ABMIL) for slide bags.

Dean Tessone's feedback on the C16 pipeline (2026-09-02), verbatim: "your
slide-level aggregation choice of top5_mean is not going to yield
interpretability as compared to ABMIL." This module is that upgrade — the
gated attention MIL of Ilse et al. 2018 (arXiv:1802.04712), the same family
the UNI paper pairs with FM embeddings (bal acc 0.957 on C16 official test).

A slide is a BAG of tile embeddings (N x D, N varies per slide). ABMIL learns
attention weights a_i over tiles and classifies the attention-weighted mean.
The attention vector is the interpretability artifact: it names WHICH tiles
drove the slide-level call, so a pathologist can look at exactly those
regions — something no fixed top-k statistic can offer.

Kept dependency-light on purpose: torch only, no torchvision, importable on
any machine with the dev venv.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GatedABMIL(nn.Module):
    """Ilse et al. gated attention: a_i ∝ w' (tanh(V h_i) ⊙ sigmoid(U h_i))."""

    def __init__(self, embed_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pre = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout))
        self.attn_v = nn.Linear(embed_dim, attn_dim)
        self.attn_u = nn.Linear(embed_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """bag: (N, D) one slide's tile embeddings -> (logit, attention (N,))."""
        h = self.pre(bag)
        scores = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h)))
        attn = torch.softmax(scores.squeeze(-1), dim=0)
        slide_repr = (attn.unsqueeze(-1) * h).sum(dim=0)
        return self.head(slide_repr).squeeze(-1), attn


class MeanPoolMIL(nn.Module):
    """Ablation twin of GatedABMIL: identical except attention is removed.

    Same LayerNorm+Dropout front end, same linear head, same training recipe —
    the bag representation is just the unweighted mean of its tiles. Comparing
    the two isolates EXACTLY what attention contributes, with no other moving
    part, which is the question Dean's top5_mean critique actually poses.

    Returns uniform "attention" so it satisfies the same interface as
    GatedABMIL (predict/attention work unchanged); a flat map is also the
    honest picture of what mean pooling attends to.
    """

    def __init__(self, embed_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pre = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout))
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.pre(bag)
        attn = torch.full((h.shape[0],), 1.0 / max(h.shape[0], 1), device=h.device)
        return self.head(h.mean(dim=0)).squeeze(-1), attn


AGGREGATORS = {"abmil": GatedABMIL, "mean": MeanPoolMIL}


def train_abmil(
    bags: list[np.ndarray],
    labels: list[int],
    embed_dim: int,
    epochs: int = 40,
    lr: float = 1e-3,  # one-bag-per-step needs this; 2e-4 stalls near uniform attention
    weight_decay: float = 1e-4,
    attn_dim: int = 128,
    dropout: float = 0.1,
    max_tiles: int = 4096,
    seed: int = 0,
    aggregator: str = "abmil",
    device: str | None = None,
    val_bags: list[np.ndarray] | None = None,
    val_labels: list[int] | None = None,
    verbose: bool = False,
) -> tuple[GatedABMIL, list[dict]]:
    """Train on (bag, slide-label) pairs; returns (model, per-epoch history).

    One slide per step (bags are ragged); bags larger than ``max_tiles`` are
    randomly subsampled per epoch during training — a standard MIL trick that
    regularizes and bounds memory. Validation always sees the full bag.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = AGGREGATORS[aggregator](embed_dim, attn_dim=attn_dim,
                                    dropout=dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos = max(sum(labels), 1)
    neg = max(len(labels) - pos, 1)
    pos_weight = torch.tensor([neg / pos], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: list[dict] = []
    order = np.arange(len(bags))
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for i in order:
            bag = bags[i]
            if bag.shape[0] > max_tiles:
                keep = rng.choice(bag.shape[0], size=max_tiles, replace=False)
                bag = bag[keep]
            x = torch.from_numpy(np.ascontiguousarray(bag, dtype=np.float32)).to(dev)
            y = torch.tensor(float(labels[i]), device=dev)
            opt.zero_grad(set_to_none=True)
            logit, _ = model(x)
            loss = loss_fn(logit.unsqueeze(0), y.unsqueeze(0))
            loss.backward()
            opt.step()
            total += float(loss.detach())
        sched.step()
        row = {"epoch": epoch, "loss": round(total / len(bags), 4)}
        if val_bags is not None and val_labels is not None:
            probs = predict(model, val_bags, device=str(dev))
            row["val_auroc"] = round(_auroc(np.array(val_labels, float), probs), 4)
        history.append(row)
        if verbose:
            print(f"[abmil] {row}", flush=True)
    return model, history


@torch.no_grad()
def predict(model: GatedABMIL, bags: list[np.ndarray],
            device: str | None = None) -> np.ndarray:
    dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.eval().to(dev)
    out = []
    for bag in bags:
        x = torch.from_numpy(np.ascontiguousarray(bag, dtype=np.float32)).to(dev)
        logit, _ = model(x)
        out.append(float(torch.sigmoid(logit)))
    return np.array(out)


@torch.no_grad()
def attention(model: GatedABMIL, bag: np.ndarray,
              device: str | None = None) -> np.ndarray:
    """Per-tile attention weights for one slide — the interpretability output."""
    dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.eval().to(dev)
    x = torch.from_numpy(np.ascontiguousarray(bag, dtype=np.float32)).to(dev)
    _, attn = model(x)
    return attn.cpu().numpy()


def _auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank AUROC with midranks for ties (kept local: no eval deps here).

    Ties are not hypothetical — sigmoid outputs saturate to exact 1.0/0.0 on
    confident slides, and a tie broken by array order is not a metric.
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    midranks = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = midranks[inverse]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
