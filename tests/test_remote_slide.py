"""remote_slide: byte-identical to the local path, and fail-closed.

The remote reader replaced full-slide downloads (which exhausted swap on an
8 GB machine). Two properties make that safe, and both are pinned here
OFFLINE against a real JPEG-tiled pyramidal TIFF served by a local HTTP Range
server — no S3, fully deterministic:

1. Identical output. Every tile equals what camelyon_lib's LOCAL path reads
   from the same file (np.array_equal), at the round-1 geometry and on the
   resize path (target mpp off the pyramid, so read_patches resamples).
2. Fail-closed. tifffile swallows OSError in places (is_jfif returns False),
   which would turn a network fault into a silently wrong decode. A failed
   range must instead raise RangeFetchError — never yield plausible tiles.
"""

from __future__ import annotations

import http.server
import socketserver
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
tiffslide = pytest.importorskip("tiffslide")
pytest.importorskip("imagecodecs")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "bench"))

import camelyon_lib as cl  # noqa: E402
import remote_slide as rs  # noqa: E402


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(rs.time, "sleep", lambda s: None)


@pytest.fixture(scope="module")
def pyramid(tmp_path_factory) -> Path:
    """A 6-level JPEG-tiled pyramid (4096 -> 128) with a pink 'tissue' blob.

    With the 0.243 um/px fallback mpp, target 0.972 picks level 2 (1024 px,
    scale 1.0) exactly like CAMELYON16, and target 0.5 picks level 1 with
    scale 0.972 — the resample path.
    """
    rng = np.random.default_rng(0)
    n = 4096
    img = np.full((n, n, 3), 244, np.uint8)
    yy, xx = np.mgrid[0:n, 0:n]
    blob = ((xx - 2300) ** 2 / 1500 ** 2 + (yy - 1900) ** 2 / 1100 ** 2) < 1
    img[blob] = (215, 110, 160)
    img = np.clip(img.astype(np.int16) + rng.integers(-18, 18, img.shape), 0, 255
                  ).astype(np.uint8)
    path = tmp_path_factory.mktemp("slide") / "slide.tif"
    levels = [img]
    while levels[-1].shape[0] > 128:
        levels.append(levels[-1][::2, ::2].copy())
    opts = dict(tile=(256, 256), compression="jpeg", photometric="rgb")
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        tif.write(levels[0], subifds=len(levels) - 1, **opts)
        for lv in levels[1:]:
            tif.write(lv, subfiletype=1, **opts)
    return path


def _serve(path: Path, fail=None):
    data = path.read_bytes()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            a, b = (int(v) for v in self.headers["Range"].split("=")[1].split("-"))
            b = min(b, len(data) - 1)
            if fail is not None and fail(a, b):
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            chunk = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/slide.tif"


def _local(path: Path, target_mpp: float, patch: int):
    slide = cl.open_slide(path)
    level, scale = cl.pick_level(slide, target_mpp=target_mpp)
    coords = cl.tissue_tiles(slide, level, patch=patch)
    tiles = list(cl.read_patches(slide, level, coords, scale, patch=patch))
    slide.close()
    return coords, level, scale, tiles


@pytest.mark.parametrize("target_mpp", [0.972, 0.5], ids=["round1-geometry", "resize-path"])
def test_remote_is_byte_identical_to_the_local_path(pyramid, target_mpp):
    coords, level, scale, tiles = _local(pyramid, target_mpp, 224)
    assert len(coords) >= 4, "fixture must yield several tissue tiles"
    if target_mpp == 0.5:
        assert abs(scale - 1) >= 0.02, "resize path not exercised"
    srv, url = _serve(pyramid)
    try:
        rep = {}
        it = rs.iter_tiles_remote(url, target_mpp=target_mpp, patch=224,
                                  strict=True, report=rep)
        r_coords, r_level, r_scale = next(it)
        r_tiles = list(it)
    finally:
        srv.shutdown()
    assert (r_level, r_scale) == (level, scale)
    assert r_coords == coords
    assert len(r_tiles) == len(tiles)
    for i, (a, b) in enumerate(zip(r_tiles, tiles)):
        assert np.array_equal(a, b), f"tile {i} at {coords[i]} differs"
    assert rep["plan_misses"] == 0          # strict=True would have raised anyway
    assert rep["fraction_of_file"] < 1.0    # never pulled the whole file


def test_a_failed_tile_range_raises_instead_of_decoding_silently(pyramid):
    """The core hazard: tifffile's is_jfif does `except OSError: return False`.
    A fault on tile bytes must surface as RangeFetchError, never as tiles."""
    tf = tifffile.TiffFile(pyramid)
    lvl2 = tf.series[0].levels[2].keyframe
    first = int(lvl2.dataoffsets[0])
    tf.close()
    srv, url = _serve(pyramid, fail=lambda a, b: a <= first <= b)
    try:
        with pytest.raises(rs.RangeFetchError):
            it = rs.iter_tiles_remote(url, target_mpp=0.972, patch=224)
            next(it)
            list(it)
    finally:
        srv.shutdown()


def test_range_fetch_error_is_not_an_oserror():
    """Closes the is_jfif path (`except OSError: return False`). A partial
    defense only — see test_poison_flag_is_load_bearing for why."""
    assert issubclass(rs.RangeFetchError, RuntimeError)
    assert not issubclass(rs.RangeFetchError, OSError)
    assert not issubclass(rs.StrictMiss, OSError)


def test_poison_flag_is_load_bearing(pyramid, monkeypatch):
    """Removing the poison check makes the reader go SILENT — this pins why it
    exists. tifffile's series parser swallows the fetch error broadly (even a
    RuntimeError), drops the pyramid level, and the pipeline then yields
    plausible tiles from the WRONG level. If this test starts failing because
    the reader raised anyway, tifffile got stricter; the check stays regardless."""
    tf = tifffile.TiffFile(pyramid)
    first = int(tf.series[0].levels[2].keyframe.dataoffsets[0])
    tf.close()
    truth = _local(pyramid, 0.972, 224)[0]
    monkeypatch.setattr(rs.HttpRangeFile, "check", lambda self, where: None)
    srv, url = _serve(pyramid, fail=lambda a, b: a <= first <= b)
    try:
        it = rs.iter_tiles_remote(url, target_mpp=0.972, patch=224)
        coords, _, _ = next(it)
        tiles = list(it)
    finally:
        srv.shutdown()
    assert tiles, "expected the unguarded reader to yield tiles silently"
    assert coords != truth, "silent output happened to be correct — hazard not shown"


def test_poison_flag_catches_a_failure_someone_swallowed(pyramid):
    srv, url = _serve(pyramid, fail=lambda a, b: a >= 50_000)
    try:
        rf = rs.HttpRangeFile(url)
        rf.check("open")                      # probe succeeded: clean
        try:
            rf._fetch(60_000, 61_000)
        except rs.RangeFetchError:
            pass                              # e.g. a library catching and moving on
        with pytest.raises(rs.RangeFetchError, match="refusing to trust"):
            rf.check("header parsing")
        rf.close()
    finally:
        srv.shutdown()


def test_strict_mode_refuses_unprefetched_reads_and_unbounded_reads(pyramid):
    srv, url = _serve(pyramid)
    try:
        rf = rs.HttpRangeFile(url)
        rf.allow_network = False
        rf.seek(100_000)
        with pytest.raises(rs.StrictMiss):
            rf.read(10)
        rf.seek(0)
        with pytest.raises(rs.StrictMiss):
            rf.read()                         # would stream the whole slide
        rf.close()
    finally:
        srv.shutdown()


def test_misaligned_response_is_rejected(pyramid, monkeypatch):
    """A 206 whose Content-Range does not match the request must not be stored."""
    data = pyramid.read_bytes()

    class Lying(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            a, b = (int(v) for v in self.headers["Range"].split("=")[1].split("-"))
            b = min(b, len(data) - 1)
            shift = 0 if a == 0 else 7        # probe honest, everything after lies
            chunk = data[a + shift:b + 1 + shift]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a + shift}-{b + shift}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Lying)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rf = rs.HttpRangeFile(f"http://127.0.0.1:{srv.server_address[1]}/slide.tif")
        with pytest.raises(rs.RangeFetchError, match="misaligned"):
            rf._fetch(10_000, 10_500)
        rf.close()
    finally:
        srv.shutdown()


def test_geometry_is_required_not_defaulted():
    """A silent geometry default is how two runs end up comparing different tiles."""
    with pytest.raises(TypeError):
        next(rs.iter_tiles_remote("http://127.0.0.1:9/x.tif"))
