"""Build industry-standard dataset manifests for the FM+ABMIL CAMELYON16 lane.

Produces data/manifests/c16_abmil_v1/:
  train.csv        slide_id,label,source_uri     (ABMIL fitting)
  val.csv          slide_id,label,source_uri     (model selection ONLY)
  test_SEALED.csv  slide_id,label,source_uri     (official C16 test — sealed)
  dataset.json     provenance, counts, split params, seal (sha256 of the test
                   manifest), query budget, and the no-peek policy

Split policy:
  - train/val is the deterministic stratified 80/20 split from
    camelyon_lib.stratified_split (same function train_abmil.py uses, so the
    registered protocol and these manifests cannot disagree);
  - test is the OFFICIAL CAMELYON16 evaluation set (reference.csv), not a
    carve-out of training. Using the community-standard test set keeps our
    number comparable to the literature (Virchow2/UNI papers report on it),
    and it is enforced one-shot by train_abmil.py --official-test.

Slide inventory comes from the CC0 S3 bucket listing + the official
reference.csv — nothing is assumed from memory. Rerunning must reproduce the
same manifests byte-for-byte; if the sealed test manifest would change, the
script refuses unless --reseal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, "scripts/bench")
from camelyon_lib import BUCKET, stratified_split  # noqa: E402

OUT = Path("data/manifests/c16_abmil_v1")
LIST_URL = ("https://camelyon-dataset.s3.us-west-2.amazonaws.com/"
            "?list-type=2&prefix=CAMELYON16/images/")


def list_bucket_slides() -> list[str]:
    """All slide stems under CAMELYON16/images/ via the S3 list API."""
    names, token = [], None
    while True:
        url = LIST_URL + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        names += [m.removesuffix(".tif").split("/")[-1]
                  for m in re.findall(r"<Key>([^<]+\.tif)</Key>", xml)]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return sorted(names)
        token = m.group(1)


def official_test_labels() -> dict[str, int]:
    """slide -> 0/1 from the official evaluation reference.csv."""
    local = Path("data/raw/CAMELYON16/evaluation/reference.csv")
    if local.exists():
        text = local.read_text()
    else:
        text = urllib.request.urlopen(f"{BUCKET}/evaluation/reference.csv",
                                      timeout=60).read().decode()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(text)
    labels = {}
    for line in text.splitlines():
        if line.startswith("test_"):
            cols = line.split(",")
            labels[cols[0].removesuffix(".tif")] = int(cols[1].strip().lower() != "normal")
    return labels


def write_manifest(path: Path, rows: list[tuple[str, int]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["slide_id", "label", "source_uri"])
    for slide, label in rows:
        w.writerow([slide, label, f"{BUCKET}/images/{slide}.tif"])
    path.write_text(buf.getvalue())
    return hashlib.sha256(buf.getvalue().encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--reseal", action="store_true",
                    help="allow the sealed test manifest to change (audit event)")
    args = ap.parse_args()

    slides = list_bucket_slides()
    train_all = [s for s in slides if s.startswith(("normal", "tumor"))]
    test_labels = official_test_labels()
    n_norm = sum(1 for s in train_all if s.startswith("normal"))
    n_tum = len(train_all) - n_norm
    print(f"[manifest] bucket: {len(train_all)} train slides "
          f"({n_norm} normal / {n_tum} tumor), "
          f"{len(test_labels)} official test slides", flush=True)

    tr, va = stratified_split(train_all, seed=args.seed, val_frac=args.val_frac)
    OUT.mkdir(parents=True, exist_ok=True)

    test_path = OUT / "test_SEALED.csv"
    test_rows = sorted(test_labels.items())
    if test_path.exists() and not args.reseal:
        old = hashlib.sha256(test_path.read_bytes()).hexdigest()
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["slide_id", "label", "source_uri"])
        for slide, label in test_rows:
            w.writerow([slide, label, f"{BUCKET}/images/{slide}.tif"])
        if hashlib.sha256(buf.getvalue().encode()).hexdigest() != old:
            raise SystemExit("[manifest] REFUSED: sealed test manifest would "
                             "change — pass --reseal only if this is a "
                             "deliberate, documented audit event")

    label = lambda s: int(s.startswith("tumor"))  # noqa: E731
    train_sha = write_manifest(OUT / "train.csv", [(s, label(s)) for s in tr])
    val_sha = write_manifest(OUT / "val.csv", [(s, label(s)) for s in va])
    test_sha = write_manifest(test_path, test_rows)

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()[:12]
    except Exception:
        sha = "unknown"
    meta = {
        "name": "c16_abmil_v1",
        "task": "slide-level metastasis detection (any tumor vs normal)",
        "source": {"bucket": BUCKET, "license": "CC0",
                   "reference": "evaluation/reference.csv (official)"},
        "counts": {"train": len(tr), "val": len(va), "test": len(test_rows),
                   "train_tumor": sum(label(s) for s in tr),
                   "val_tumor": sum(label(s) for s in va),
                   "test_tumor": sum(l for _, l in test_rows)},
        "split": {"method": "camelyon_lib.stratified_split", "seed": args.seed,
                  "val_frac": args.val_frac,
                  "test": "official CAMELYON16 evaluation set (not a carve-out)"},
        "sha256": {"train.csv": train_sha, "val.csv": val_sha,
                   "test_SEALED.csv": test_sha},
        "seal_policy": {
            "query_budget": 1,
            "enforced_by": "scripts/bench/train_abmil.py --official-test "
                           "(refuses without a registered protocol; refuses a "
                           "second run)",
            "statement": "No model, hyperparameter choice, or aggregation may "
                         "be informed by test slides. Test embeddings are "
                         "computed blind and scored once."},
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": sha,
    }
    (OUT / "dataset.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[manifest] wrote {OUT}/: train={len(tr)} val={len(va)} "
          f"test={len(test_rows)} (sealed, sha256={test_sha[:12]}…)", flush=True)


if __name__ == "__main__":
    main()
