"""camelyon_lib: the streaming toolkit's geometry and download guards.

read_patches must request MORE level pixels when the level is finer than the
target mpp (then downsample), and fetch must never leave a non-TIFF or
truncated file where a resumed run would trust it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "bench"))

import camelyon_lib as cl  # noqa: E402


class _Slide:
    level_downsamples = [1.0, 2.0]

    def __init__(self):
        self.requested = []

    def read_region(self, loc, level, size):
        from PIL import Image
        self.requested.append(size)
        return Image.new("RGB", size)


def test_read_patches_requests_more_pixels_from_a_finer_level():
    pytest.importorskip("PIL")
    s = _Slide()
    tiles = list(cl.read_patches(s, 1, [(0, 0)], scale=0.5, patch=224))
    assert s.requested == [(448, 448)]
    assert tiles[0].shape == (224, 224, 3)
    s = _Slide()
    list(cl.read_patches(s, 1, [(0, 0)], scale=1.0, patch=96))
    assert s.requested == [(96, 96)]


def test_stratified_split_deterministic_and_disjoint():
    names = [f"normal_{i:03d}" for i in range(20)] + [f"tumor_{i:03d}" for i in range(10)]
    tr, va = cl.stratified_split(names, seed=0)
    assert (tr, va) == cl.stratified_split(names, seed=0)
    assert not set(tr) & set(va) and set(tr) | set(va) == set(names)
    assert sum(n.startswith("tumor") for n in va) == 2 and len(va) == 6
    assert cl.stratified_split(names, seed=1)[1] != va


def _fake_curl(body: bytes):
    def run(cmd, **kw):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(body)
    return run


TIFF = b"II*\x00" + b"\x00" * 96  # 100 bytes


def test_fetch_rejects_an_s3_error_body(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "remote_size", lambda url: 100)
    monkeypatch.setattr(cl.subprocess, "run", _fake_curl(b"<Error><Code>AccessDenied</Code></Error>"))
    with pytest.raises(RuntimeError):
        cl.fetch("images/x.tif", tmp_path / "x.tif")
    assert not (tmp_path / "x.tif").exists()
    assert not (tmp_path / "x.tif.part").exists()


def test_fetch_rejects_a_truncated_download(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "remote_size", lambda url: 100)
    monkeypatch.setattr(cl.subprocess, "run", _fake_curl(TIFF[:50]))
    with pytest.raises(RuntimeError):
        cl.fetch("images/x.tif", tmp_path / "x.tif")
    assert not (tmp_path / "x.tif").exists()


def test_fetch_replaces_a_corrupt_cached_slide(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "remote_size", lambda url: 100)
    monkeypatch.setattr(cl.subprocess, "run", _fake_curl(TIFF))
    dest = tmp_path / "x.tif"
    dest.write_bytes(b"<Error/>")  # what an older curl -sL run left behind
    assert cl.fetch("images/x.tif", dest) == dest
    assert dest.read_bytes() == TIFF
    assert not (tmp_path / "x.tif.part").exists()


def test_fetch_trusts_a_complete_part_without_redownloading(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "remote_size", lambda url: 100)
    calls = []
    monkeypatch.setattr(cl.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    (tmp_path / "x.tif.part").write_bytes(TIFF)  # a finished prefetch
    assert cl.fetch("images/x.tif", tmp_path / "x.tif").read_bytes() == TIFF
    assert calls == []


def test_looks_complete_requires_tiff_magic_only_for_tif(tmp_path):
    p = tmp_path / "reference.csv"
    p.write_bytes(b"image,type\n")
    assert cl.looks_complete(p, None)
    t = tmp_path / "x.tif"
    t.write_bytes(b"image,type\n")
    assert not cl.looks_complete(t, None)
    t.write_bytes(np.zeros(8, np.uint8).tobytes().replace(b"\x00\x00\x00\x00", b"MM\x00*", 1))
    assert cl.looks_complete(t, 8) and not cl.looks_complete(t, 9)
