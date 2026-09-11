"""Train gated ABMIL on CAMELYON16 FM bags — with the sealed-test discipline.

Pipeline position: embed_c16_fm.py wrote one bag per slide
(runs/c16/embeds/<run_key>/{slide}.npz). This script:

  1. trains ABMIL on the TRAIN manifest slides and reports on the VAL
     manifest slides (data/manifests/c16_abmil_v1/, each CSV sha256-checked
     against dataset.json). The split is the committed manifest — never
     inferred from whichever bags happen to exist — and --seed changes only
     model init and shuffling, so val mean +/- std across seeds is a real
     seed-variance number on one fixed val set;
  2. pre-registers the protocol (hyperparams + val AUROC + git sha) to
     protocol.json BEFORE any test evaluation;
  3. evaluates the official 129-slide test set (--official-test) under the
     manifest's query budget: a GLOBAL ledger
     (data/manifests/c16_abmil_v1/official_queries.jsonl) is charged before
     scoring, independent of encoder, seed or output dir — another seed is
     not another shot;
  4. exports per-slide attention (tile coords + weights) — the
     interpretability artifact Dean contrasted with top5_mean: it names WHICH
     regions drove each slide-level call. --render draws the overlay PNG.

Outputs land under runs/c16/abmil/<run_key>/ (gitignored) and the
committable parts are mirrored to results/c16_abmil/<run_key>/ (protocol,
aggregate official result, attention PNGs).

The C16 official test was already consumed once by the PCam pipeline
(0.827 with top5_mean). This lane re-uses it for ONE pre-registered ABMIL
evaluation; the comparison to 0.827 is the headline deliverable.

Usage (training machine):
  python scripts/bench/train_abmil.py --embeds runs/c16/embeds/phikon_mpp0.972 --seed 0
  python scripts/bench/train_abmil.py --embeds ... --seed S --official-test
  python scripts/bench/train_abmil.py --embeds ... --seed S --render test_001
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/bench")
import torch  # noqa: E402

from oncoscope.models.abmil import (  # noqa: E402
    AGGREGATORS, _auroc, attention, predict, train_abmil,
)

MANIFESTS = Path("data/manifests/c16_abmil_v1")
RESULTS = Path("results/c16_abmil")
MANIFEST_FILES = ("train.csv", "val.csv", "test_SEALED.csv")
LEDGER = "official_queries.jsonl"


def load_manifests(mdir: Path):
    """dataset.json plus the three manifests, each sha256-checked against the seal.

    A mismatch is refused: an edited manifest is exactly what the seal exists
    to catch, and every downstream number cites these hashes.
    """
    meta_path = mdir / "dataset.json"
    if not meta_path.exists():
        raise SystemExit(f"[abmil] missing {meta_path} — run scripts/build_c16_manifests.py")
    meta = json.loads(meta_path.read_text())
    splits = {}
    for fname in MANIFEST_FILES:
        p = mdir / fname
        if not p.exists():
            raise SystemExit(f"[abmil] missing manifest {p}")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha != meta["sha256"][fname]:
            raise SystemExit(f"[abmil] REFUSED: {p} sha256 {sha[:12]} != sealed "
                             f"{meta['sha256'][fname][:12]} — manifest edited or corrupt")
        with p.open(newline="") as fh:
            splits[fname] = [(r["slide_id"], int(r["label"])) for r in csv.DictReader(fh)]
    return meta, splits


def require_bags(embed_dir: Path, names: list[str], what: str, hint: str) -> None:
    missing = [n for n in names if not (embed_dir / f"{n}.npz").exists()]
    if missing:
        raise SystemExit(f"[abmil] REFUSED: {len(missing)} {what} slides have no bag in "
                         f"{embed_dir} (first: {missing[:3]}) — {hint}")


class LazyBags:
    """Bags read from their .npz on each access, never all held in RAM.

    Training touches every bag every epoch, so holding them resident keeps the
    whole set in memory at once: 3.2 GB of imagenet_resnet float16 on an 8 GB
    machine, which macOS compressed and then thrashed decompressing each epoch
    (trainer at 0% CPU, swap grown onto a 99%-full disk). Reading on demand
    keeps one bag resident and uses the .npz files already on disk, so it costs
    no disk space; decompressing is ~10 ms against ~57 ms of compute per bag.
    Bags stay float16 as stored — every consumer converts to float32 at use,
    and fp16 -> fp32 is exact — so results are bit-identical to holding them.
    """

    def __init__(self, embed_dir: Path, names: list[str]):
        self._paths = [embed_dir / f"{n}.npz" for n in names]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, i) -> np.ndarray:
        with np.load(self._paths[i]) as z:
            return np.asarray(z["embeddings"])

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_bags(embed_dir: Path, names: list[str]) -> LazyBags:
    return LazyBags(embed_dir, names)


def queries_spent(mdir: Path) -> int:
    p = mdir / LEDGER
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip())


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()[:12]
    except Exception:
        return "unknown"


def export_attention(model, embed_dir: Path, names: list[str], out_dir: Path,
                     device: str, top_k: int = 200) -> None:
    att_dir = out_dir / "attention"
    att_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        with np.load(embed_dir / f"{n}.npz") as z:
            bag = z["embeddings"].astype(np.float32)
            coords, level, scale, patch = z["coords"], z["level"], z["scale"], z["patch"]
        a = attention(model, bag, device=device)
        top = np.argsort(a)[::-1][:top_k]
        np.savez_compressed(att_dir / f"{n}.npz",
                            attn=a[top].astype(np.float32),
                            coords=coords[top], level=level, scale=scale,
                            patch=patch, n_tiles=len(a))


def render(name: str, out_dir: Path, mirror: Path) -> None:
    """Overlay exported attention on the slide thumbnail (fetches the slide)."""
    from PIL import Image, ImageDraw
    from camelyon_lib import fetch, open_slide
    with np.load(out_dir / "attention" / f"{name}.npz") as z:
        attn, coords, level, patch = z["attn"], z["coords"], int(z["level"]), int(z["patch"])
    slide_path = Path("runs/c16/slides") / f"{name}.tif"
    fetch(f"images/{name}.tif", slide_path)
    slide = open_slide(slide_path)
    thumb_level = len(slide.level_downsamples) - 1
    tw, th = slide.level_dimensions[thumb_level]
    img = slide.read_region((0, 0), thumb_level, (tw, th)).convert("RGB")
    W, H = slide.level_dimensions[level]
    fx, fy = tw / W, th / H
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    lo, hi = float(attn.min()), float(attn.max())
    for (x, y), a in zip(coords, attn):
        heat = (a - lo) / (hi - lo + 1e-9)
        draw.rectangle([x * fx, y * fy, (x + patch) * fx, (y + patch) * fy],
                       fill=(255, int(80 * (1 - heat)), 0, int(40 + 180 * heat)))
    slide.close()
    dest = mirror / "attention" / f"{name}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB").save(dest)
    print(f"[abmil] wrote {dest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeds", required=True,
                    help="bag directory from embed_c16_fm.py, e.g. runs/c16/embeds/phikon_mpp0.972")
    ap.add_argument("--manifests", default=str(MANIFESTS),
                    help="sealed manifest directory (train.csv/val.csv/test_SEALED.csv/dataset.json)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated slide ids deliberately dropped (recorded in protocol.json)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--attn-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-tiles", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0,
                    help="model init + shuffling only; the split is the manifest")
    ap.add_argument("--aggregator", default="abmil", choices=sorted(AGGREGATORS),
                    help="abmil = gated attention; mean = unweighted mean pooling, "
                         "the ablation that isolates what attention buys")
    ap.add_argument("--device", default=None)
    ap.add_argument("--official-test", action="store_true",
                    help="score the official test set under the manifest's query budget")
    ap.add_argument("--render", default="",
                    help="slide name: draw attention overlay PNG and exit")
    args = ap.parse_args()

    embed_dir = Path(args.embeds)
    # seed-suffixed run key so multi-seed runs never clobber each other;
    # seed 0 keeps the bare key. The one-shot guard does NOT live here — it is
    # the manifest ledger below, shared by every run key.
    run_key = (embed_dir.name
               + ("" if args.aggregator == "abmil" else f"_{args.aggregator}")
               + (f"_seed{args.seed}" if args.seed else ""))
    out_dir = Path("runs/c16/abmil") / run_key
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror = RESULTS / run_key
    mdir = Path(args.manifests)
    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.render:
        render(args.render, out_dir, mirror)
        return

    meta, splits = load_manifests(mdir)
    protocol_path = out_dir / "protocol.json"
    model_path = out_dir / "model.pt"

    if args.official_test:
        if not protocol_path.exists():
            raise SystemExit("[abmil] REFUSED: no protocol.json — train first; the "
                             "protocol must be registered before any test eval")
        official = out_dir / "official_test.json"
        if official.exists():
            raise SystemExit(f"[abmil] REFUSED: {official} exists — the official "
                             "test is one-shot; a second run is not reportable")
        ledger = mdir / LEDGER
        budget = int(meta["seal_policy"]["query_budget"])
        spent = queries_spent(mdir)
        if spent >= budget:
            raise SystemExit(f"[abmil] REFUSED: the sealed C16 test budget is spent "
                             f"({spent}/{budget}, see {ledger}). Another seed or "
                             "encoder is not another shot; a further query is a "
                             "governance decision recorded in dataset.json, not a rerun")
        protocol = json.loads(protocol_path.read_text())
        test_rows = splits["test_SEALED.csv"]
        names = [n for n, _ in test_rows]
        labels = dict(test_rows)
        require_bags(embed_dir, names, "test",
                     "the official eval must cover every sealed test slide; "
                     "finish embed_c16_fm.py on the test list. Excluding a "
                     "sealed test slide is not an option")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        model = AGGREGATORS[ckpt.get("aggregator", "abmil")](
            ckpt["embed_dim"], attn_dim=ckpt["attn_dim"], dropout=ckpt["dropout"])
        model.load_state_dict(ckpt["state"])
        bags = load_bags(embed_dir, names)
        # Everything fallible is done. Charge the ledger BEFORE scoring so a
        # crash from here on can never masquerade as an unspent shot.
        with ledger.open("a") as fh:
            fh.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "run_key": run_key,
                "seed": args.seed, "embeds": str(embed_dir),
                "protocol_val_auroc": protocol["val_auroc"], "n_slides": len(names),
                "git_sha": git_sha()}) + "\n")
        probs = predict(model, bags, device=dev)
        y = np.array([labels[n] for n in names], float)
        auroc = _auroc(y, probs)
        rec = {"run_key": run_key, "seed": args.seed, "n_slides": len(names),
               "auroc": round(float(auroc), 4),
               "protocol_val_auroc": protocol["val_auroc"],
               "baseline_pcam_top5_mean": 0.827,
               "query": f"{spent + 1}/{budget}",
               "test_manifest_sha256": meta["sha256"]["test_SEALED.csv"],
               "git_sha": git_sha(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        official.write_text(json.dumps(
            {**rec, "per_slide": {n: round(float(p), 4) for n, p in zip(names, probs)}},
            indent=1))
        # committed mirror carries aggregates only — per-slide test scores are
        # how a sealed set gets tuned against by hand
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / "official_test.json").write_text(json.dumps(rec, indent=1))
        shutil.copy(protocol_path, mirror / "protocol.json")
        export_attention(model, embed_dir, names, out_dir, dev)
        print(f"[abmil] OFFICIAL C16 TEST: AUROC = {auroc:.4f} "
              f"(PCam+top5_mean baseline: 0.827) -> {official}", flush=True)
        print(f"[abmil] the shot is spent: commit {mirror}/ and {ledger}", flush=True)
        return

    # ---- training path ----
    excluded = sorted({n for n in args.exclude.split(",") if n})
    tr_rows = [(n, l) for n, l in splits["train.csv"] if n not in excluded]
    va_rows = [(n, l) for n, l in splits["val.csv"] if n not in excluded]
    tr, va = [n for n, _ in tr_rows], [n for n, _ in va_rows]
    require_bags(embed_dir, tr + va, "train/val",
                 "finish embed_c16_fm.py first, or record a deliberate "
                 "exclusion with --exclude")
    print(f"[abmil] {run_key}: {len(tr)} train / {len(va)} val slides "
          f"(manifests {mdir}, {len(excluded)} excluded)", flush=True)
    tr_bags, va_bags = load_bags(embed_dir, tr), load_bags(embed_dir, va)
    embed_dim = tr_bags[0].shape[1]
    model, history = train_abmil(
        tr_bags, [l for _, l in tr_rows], embed_dim=embed_dim,
        epochs=args.epochs, lr=args.lr, attn_dim=args.attn_dim,
        dropout=args.dropout, max_tiles=args.max_tiles, seed=args.seed,
        aggregator=args.aggregator,
        device=dev, val_bags=va_bags, val_labels=[l for _, l in va_rows],
        verbose=True)
    val_auroc = history[-1]["val_auroc"]
    torch.save({"state": model.state_dict(), "embed_dim": embed_dim,
                "attn_dim": args.attn_dim, "dropout": args.dropout,
                "aggregator": args.aggregator}, model_path)
    protocol = {
        "run_key": run_key, "embed_dim": embed_dim,
        "hyperparams": {k: getattr(args, k) for k in
                        ("epochs", "lr", "attn_dim", "dropout", "max_tiles",
                         "seed", "aggregator")},
        "split": {"manifests": str(mdir), "sha256": meta["sha256"],
                  "train": tr, "val": va, "excluded": excluded},
        "model_selection": "last epoch of the cosine schedule; val AUROC is "
                           "not epoch-selected",
        "val_auroc": val_auroc, "history": history,
        "git_sha": git_sha(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "declaration": "Registered before official test. The C16 official test "
                       "will be evaluated ONCE with this model; result reported "
                       "regardless of outcome, vs the 0.827 PCam+top5_mean baseline.",
    }
    protocol_path.write_text(json.dumps(protocol, indent=1))
    mirror.mkdir(parents=True, exist_ok=True)
    shutil.copy(protocol_path, mirror / "protocol.json")
    export_attention(model, embed_dir, va, out_dir, dev)
    print(f"[abmil] val AUROC = {val_auroc} -> {protocol_path}", flush=True)
    print("[abmil] next: --official-test (one shot) after embedding test slides",
          flush=True)


if __name__ == "__main__":
    main()
