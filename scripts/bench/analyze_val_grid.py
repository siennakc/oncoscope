"""Aggregate the FM+ABMIL validation grid into the 2x2, with paired bootstrap CIs.

The grid ({phikon, imagenet_resnet} x {abmil, mean} x seeds 0-2) is scored on
the 54-slide VAL split only; nothing here reads the sealed test manifest.

The trainer records only each run's aggregate val AUROC, but comparing
conditions honestly needs per-slide scores: 54 slides (22 tumor) make a lone
AUROC noisy, so every contrast is a PAIRED, class-stratified bootstrap of the
difference, resampling the same slides for both arms. Per-slide scores are
rebuilt from each run's saved model.pt and must reproduce the logged val AUROC
before they are used — a mismatch means the reconstruction is wrong, and the
script refuses rather than reporting numbers built on it.

Contrasts use each condition's SEED-AVERAGED score per slide, so seed noise is
averaged out of the comparison rather than cherry-picked.

Usage:  python scripts/bench/analyze_val_grid.py
Writes: results/c16_abmil/val_grid.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/bench")
import torch  # noqa: E402

from oncoscope.models.abmil import AGGREGATORS, _auroc, predict  # noqa: E402
from train_abmil import MANIFESTS, load_bags, load_manifests  # noqa: E402

ENCODERS = ("phikon", "imagenet_resnet")
AGGREGATORS_RUN = ("abmil", "mean")
SEEDS = (0, 1, 2)
MPP = "0.972"
B = 10_000
OUT = Path("results/c16_abmil/val_grid.json")


def run_key(enc: str, agg: str, seed: int) -> str:
    return (f"{enc}_mpp{MPP}" + ("" if agg == "abmil" else f"_{agg}")
            + (f"_seed{seed}" if seed else ""))


def stratified_indices(y: np.ndarray, rng, n_boot: int) -> np.ndarray:
    """Resample positives and negatives separately: every replicate keeps the
    val set's class balance, so none is degenerate and AUROC is always defined."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, (n_boot, len(pos))),
                           rng.choice(neg, (n_boot, len(neg)))], axis=1)


def ci(values) -> list[float]:
    return [round(float(v), 4) for v in np.percentile(values, [2.5, 97.5])]


def main() -> None:
    _, splits = load_manifests(MANIFESTS)
    val = splits["val.csv"]
    names = [n for n, _ in val]
    y = np.array([l for _, l in val], float)

    per_slide: dict[tuple, np.ndarray] = {}
    per_run: dict[str, float] = {}
    for enc in ENCODERS:
        bags = load_bags(Path(f"runs/c16/embeds/{enc}_mpp{MPP}"), names)
        for agg in AGGREGATORS_RUN:
            for s in SEEDS:
                key = run_key(enc, agg, s)
                d = Path("runs/c16/abmil") / key
                proto = json.loads((d / "protocol.json").read_text())
                if proto["split"]["val"] != names:
                    raise SystemExit(f"[grid] REFUSED: {key} was not scored on the sealed val manifest")
                ck = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
                if ck.get("aggregator", "abmil") != agg:
                    raise SystemExit(f"[grid] REFUSED: {key} checkpoint is {ck.get('aggregator')}, expected {agg}")
                model = AGGREGATORS[agg](ck["embed_dim"], attn_dim=ck["attn_dim"],
                                         dropout=ck["dropout"])
                model.load_state_dict(ck["state"])
                p = predict(model, bags, device="cpu")
                got = _auroc(y, p)
                if abs(got - proto["val_auroc"]) > 1e-4:      # logged value is rounded to 4 dp
                    raise SystemExit(f"[grid] REFUSED: {key} rebuilt val AUROC {got:.4f} != logged "
                                     f"{proto['val_auroc']} — per-slide scores are not trustworthy")
                per_slide[(enc, agg, s)] = p
                per_run[key] = round(float(got), 4)
        del bags

    rng = np.random.default_rng(0)
    idx = stratified_indices(y, rng, B)
    ens = {(e, a): np.mean([per_slide[(e, a, s)] for s in SEEDS], axis=0)
           for e in ENCODERS for a in AGGREGATORS_RUN}
    boot = {c: np.array([_auroc(y[i], ens[c][i]) for i in idx]) for c in ens}

    cells = {}
    for e in ENCODERS:
        for a in AGGREGATORS_RUN:
            seeds = [per_run[run_key(e, a, s)] for s in SEEDS]
            cells[f"{e}+{a}"] = {
                "seed_aurocs": seeds,
                "seed_mean": round(float(np.mean(seeds)), 4),
                "seed_std": round(float(np.std(seeds, ddof=1)), 4),
                "seed_ensemble_auroc": round(float(_auroc(y, ens[(e, a)])), 4),
                "seed_ensemble_ci95": ci(boot[(e, a)]),
            }

    def contrast(label, a, b):
        diff = boot[a] - boot[b]
        return {"contrast": label, "delta": round(float(_auroc(y, ens[a]) - _auroc(y, ens[b])), 4),
                "ci95": ci(diff), "p_delta_le_0": round(float((diff <= 0).mean()), 4)}

    P, I = "phikon", "imagenet_resnet"
    contrasts = [
        contrast("FM effect, ABMIL (phikon - imagenet)", (P, "abmil"), (I, "abmil")),
        contrast("FM effect, mean-pool (phikon - imagenet)", (P, "mean"), (I, "mean")),
        contrast("attention effect, phikon (abmil - mean)", (P, "abmil"), (P, "mean")),
        contrast("attention effect, imagenet (abmil - mean)", (I, "abmil"), (I, "mean")),
    ]
    inter = (boot[(P, "abmil")] - boot[(P, "mean")]) - (boot[(I, "abmil")] - boot[(I, "mean")])
    contrasts.append({"contrast": "interaction (attention gain on phikon - on imagenet)",
                      "delta": round(float((_auroc(y, ens[(P, "abmil")]) - _auroc(y, ens[(P, "mean")]))
                                           - (_auroc(y, ens[(I, "abmil")]) - _auroc(y, ens[(I, "mean")]))), 4),
                      "ci95": ci(inter), "p_delta_le_0": round(float((inter <= 0).mean()), 4)})

    best = max(cells, key=lambda c: cells[c]["seed_mean"])
    rec = {
        "split": "val (54 slides, 22 tumor) — sealed manifest; test NOT touched",
        "geometry": f"224px @ {MPP} um/px",
        "bootstrap": f"{B} class-stratified paired resamples of val slides; contrasts on seed-averaged scores",
        "model_selection": "last epoch of the cosine schedule (not epoch-selected)",
        "cells": cells, "contrasts": contrasts, "per_run": per_run,
        "best_condition_by_seed_mean": best,
        "caveats": [
            "54 val slides is small: CIs are wide and a val ranking is not a test result.",
            "imagenet_resnet is a GENERIC encoder, weaker than the lymph-node-trained PCam "
            "ResNet behind the 0.827 baseline, so the FM effect here is measured against a "
            "flattering control.",
            "Choosing the best of several conditions on val makes that condition's val AUROC "
            "optimistic; only the sealed test is unbiased.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))

    print(f"\n{'condition':<24}{'seeds (val AUROC)':<28}{'mean ± std':<16}{'seed-ensemble [95% CI]'}")
    for c, v in cells.items():
        print(f"{c:<24}{', '.join(f'{x:.3f}' for x in v['seed_aurocs']):<28}"
              f"{v['seed_mean']:.3f} ± {v['seed_std']:.3f}   "
              f"{v['seed_ensemble_auroc']:.3f} [{v['seed_ensemble_ci95'][0]:.3f}, {v['seed_ensemble_ci95'][1]:.3f}]")
    print()
    for c in contrasts:
        print(f"{c['contrast']:<52} Δ {c['delta']:+.3f}  95% CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
              f"  P(Δ≤0)={c['p_delta_le_0']:.3f}")
    print(f"\nbest condition by seed mean: {best}   -> {OUT}")


if __name__ == "__main__":
    main()
