"""End-to-end test of scripts/bench/train_abmil.py on synthetic bags.

Exercises the parts unit tests can't: the manifest-driven, sha-sealed split,
protocol pre-registration, attention export, and — most importantly — the
official-test gates: refuse without a protocol, refuse on missing test bags,
refuse a second run under the same key, and refuse a second run under ANY
other key (another seed) once the manifest's query budget is spent. Runs in
a temp cwd so no repo state is touched.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bench" / "train_abmil.py"
DIM = 16
EMBEDS = "runs/c16/embeds/fake_mpp0.5"
MANIFESTS = "data/manifests/c16_abmil_v1"

_DIR = np.random.default_rng(7).normal(0, 1, DIM)
_DIR = (_DIR - _DIR.mean()) / np.linalg.norm(_DIR)


def _write_bag(path: Path, positive: bool, rng, n=30):
    bag = rng.normal(0, 1, (n, DIM)).astype(np.float32)
    if positive:
        idx = rng.choice(n, size=3, replace=False)
        bag[idx] += (8.0 * _DIR).astype(np.float32)
    coords = rng.integers(0, 5000, (n, 2)).astype(np.int32)
    np.savez_compressed(path, embeddings=bag.astype(np.float16), coords=coords,
                        level=2, scale=1.0, patch=224, target_mpp=0.5,
                        model="fake", repo="fake/fake")


def _manifest_csv(rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["slide_id", "label", "source_uri"])
    for slide, label in rows:
        w.writerow([slide, label, f"https://example.invalid/{slide}.tif"])
    return buf.getvalue()


def _write_manifests(cwd: Path, train, val, test, budget=1) -> Path:
    mdir = cwd / MANIFESTS
    mdir.mkdir(parents=True, exist_ok=True)
    shas = {}
    for fname, rows in (("train.csv", train), ("val.csv", val), ("test_SEALED.csv", test)):
        text = _manifest_csv(rows)
        (mdir / fname).write_text(text)
        shas[fname] = hashlib.sha256(text.encode()).hexdigest()
    (mdir / "dataset.json").write_text(json.dumps(
        {"name": "fake", "sha256": shas, "seal_policy": {"query_budget": budget}}))
    return mdir


def _run(cwd: Path, *extra, expect_fail=False):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--embeds", EMBEDS,
         "--epochs", "60", "--device", "cpu", *extra],
        cwd=cwd, capture_output=True, text=True,
        env={"PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts' / 'bench'}",
             "PATH": "/usr/bin:/bin"})
    if expect_fail:
        assert proc.returncode != 0, f"expected refusal, got:\n{proc.stdout}"
    else:
        assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc


def _protocol(cwd: Path, run_key: str) -> dict:
    return json.loads((cwd / "runs/c16/abmil" / run_key / "protocol.json").read_text())


def _ledger_lines(cwd: Path) -> list[str]:
    p = cwd / MANIFESTS / "official_queries.jsonl"
    return [l for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    cwd = tmp_path_factory.mktemp("abmil_e2e")
    embed_dir = cwd / EMBEDS
    embed_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    train, val = [], []
    for i in range(12):
        for prefix, positive in (("tumor", True), ("normal", False)):
            name = f"{prefix}_{i:03d}"
            _write_bag(embed_dir / f"{name}.npz", positive, rng)
            (val if i >= 8 else train).append((name, int(positive)))  # 16 / 8
    test = [(f"test_{i:03d}", int(i % 2 == 0)) for i in range(6)]
    _write_manifests(cwd, train, val, test)
    return cwd


def test_official_refused_before_training(world):
    proc = _run(world, "--official-test", expect_fail=True)
    assert "no protocol.json" in proc.stderr + proc.stdout


def test_training_uses_the_sealed_manifest_split(world):
    _run(world)
    protocol = _protocol(world, "fake_mpp0.5")
    assert protocol["val_auroc"] >= 0.9, protocol["history"][-3:]
    assert "declaration" in protocol and "ONCE" in protocol["declaration"]
    mdir = world / MANIFESTS
    with (mdir / "val.csv").open(newline="") as fh:
        manifest_val = [r["slide_id"] for r in csv.DictReader(fh)]
    assert protocol["split"]["val"] == manifest_val
    assert len(protocol["split"]["train"]) == 16
    assert not set(protocol["split"]["train"]) & set(protocol["split"]["val"])
    seal = json.loads((mdir / "dataset.json").read_text())["sha256"]
    assert protocol["split"]["sha256"] == seal
    # runs/ is gitignored: the protocol is mirrored where it can be committed
    assert (world / "results/c16_abmil/fake_mpp0.5/protocol.json").exists()
    # attention exported for every val slide, weights sorted descending
    out = world / "runs/c16/abmil/fake_mpp0.5"
    for name in protocol["split"]["val"]:
        with np.load(out / "attention" / f"{name}.npz") as z:
            attn = z["attn"]
            assert (np.diff(attn) <= 1e-9).all()
            assert z["coords"].shape == (len(attn), 2)


def test_seed_changes_the_model_not_the_split(world):
    _run(world, "--seed", "1")
    p0, p1 = _protocol(world, "fake_mpp0.5"), _protocol(world, "fake_mpp0.5_seed1")
    assert p1["split"]["train"] == p0["split"]["train"]
    assert p1["split"]["val"] == p0["split"]["val"]
    assert p1["hyperparams"]["seed"] == 1


def test_missing_train_bag_is_refused_unless_excluded(world):
    bag = world / EMBEDS / "normal_000.npz"
    backup = bag.read_bytes()
    bag.unlink()
    try:
        proc = _run(world, "--seed", "2", expect_fail=True)
        assert "no bag" in proc.stderr + proc.stdout
        _run(world, "--seed", "2", "--exclude", "normal_000")
        p = _protocol(world, "fake_mpp0.5_seed2")
        assert p["split"]["excluded"] == ["normal_000"]
        assert "normal_000" not in p["split"]["train"]
    finally:
        bag.write_bytes(backup)


def test_tampered_manifest_is_refused(world):
    val_csv = world / MANIFESTS / "val.csv"
    orig = val_csv.read_bytes()
    val_csv.write_bytes(orig + b"tumor_000,1,x\n")
    try:
        proc = _run(world, "--seed", "3", expect_fail=True)
        assert "sha256" in proc.stderr + proc.stdout
    finally:
        val_csv.write_bytes(orig)


def test_official_refused_when_test_slides_missing(world):
    rng = np.random.default_rng(42)
    embed_dir = world / EMBEDS
    for i in range(4):  # leave two unembedded
        _write_bag(embed_dir / f"test_{i:03d}.npz", i % 2 == 0, rng)
    proc = _run(world, "--official-test", expect_fail=True)
    assert "no bag" in proc.stderr + proc.stdout
    assert _ledger_lines(world) == []  # nothing charged for a refusal


def test_official_runs_once_then_refuses_under_every_key(world):
    rng = np.random.default_rng(43)
    embed_dir = world / EMBEDS
    for i in range(4, 6):
        _write_bag(embed_dir / f"test_{i:03d}.npz", i % 2 == 0, rng)
    proc = _run(world, "--official-test")
    assert "OFFICIAL C16 TEST" in proc.stdout
    out = world / "runs/c16/abmil/fake_mpp0.5"
    rec = json.loads((out / "official_test.json").read_text())
    assert rec["n_slides"] == 6
    # 60 quick epochs on 16 tiny bags: demand clear separation, not perfection
    assert rec["auroc"] >= 0.85, rec["auroc"]
    assert rec["baseline_pcam_top5_mean"] == 0.827
    assert rec["query"] == "1/1"
    lines = _ledger_lines(world)
    assert len(lines) == 1 and json.loads(lines[0])["run_key"] == "fake_mpp0.5"
    mirror = json.loads((world / "results/c16_abmil/fake_mpp0.5/official_test.json").read_text())
    assert mirror["auroc"] == rec["auroc"]
    assert "per_slide" in rec and "per_slide" not in mirror
    assert (out / "attention" / "test_000.npz").exists()
    # same key again
    proc = _run(world, "--official-test", expect_fail=True)
    assert "one-shot" in proc.stderr + proc.stdout
    # another seed has its own registered protocol and no official_test.json —
    # the ledger, not the run key, is what makes the shot one-shot
    proc = _run(world, "--seed", "1", "--official-test", expect_fail=True)
    assert "budget" in proc.stderr + proc.stdout
    assert len(_ledger_lines(world)) == 1
