"""Read CAMELYON16 tiles straight from S3 — only the bytes the pipeline uses.

Why: a slide is a ~2 GB BigTIFF pyramid, but the embedding pipeline reads
exactly one pyramid level (plus the tiny smallest level for its tissue mask)
— about 1-5% of the file. Downloading the whole object held 2-4 GB on disk
per slide, which on an 8 GB-RAM, ~96%-full laptop exhausted swap and stalled
the job for hours (2026-09-10). This module fetches only the TIFF tiles the
pipeline actually reads, concurrently, into memory. Zero slide bytes touch
disk; measured ~1-5% of each file over the network.

Correctness is structural, not re-implemented. camelyon_lib.pick_level,
tissue_tiles and read_patches are called UNCHANGED on a real tiffslide.TiffSlide
— the only difference is that its file object serves bytes from memory
instead of disk. So the tiling semantics cannot drift from the local path.
Verified pixel-exact (np.array_equal on every tile, max diff 0) against the
local path on normal_006 @0.972 (360/360), tumor_012 @0.972 (1727/1727) and
normal_006 @0.5 — the resize path — (1360/1360), plus slide-edge regions.

Fail-closed, and the POISON FLAG is what makes it so. tifffile recovers from
read errors in ways that turn a network fault into a silently WRONG decode:
its series parser catches exceptions broadly ("corrupted tag list") and drops
the pyramid level, after which pick_level picks a different level and the
pipeline yields plausible, wrong tiles. The exception TYPE cannot prevent that
— mutation testing showed a RuntimeError is swallowed just the same and 30
wrong tiles come out. So every fetch failure sets a poison flag, checked after
header parsing, after prefetch and after decoding; the embedder writes a bag
only when the slide finishes clean. Nothing a library swallows can reach disk.
RangeFetchError is additionally not an OSError, which closes the narrower
is_jfif path (`except OSError: return False`) even before the flag is checked.
tests/test_remote_slide.py pins both, including that removing the flag makes
the reader go silent — do not remove it on the theory that the exception type
suffices.

Planning which tiles to prefetch mirrors read_patches' region geometry. That
mirror is an OPTIMIZATION only: if it ever drifts, the missed reads are
fetched on demand (strict=False, the production mode) and output stays
byte-identical; the report's `plan_misses` makes such drift visible.
"""
from __future__ import annotations

import bisect
import io
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

from camelyon_lib import pick_level, read_patches, tissue_tiles


class RangeFetchError(RuntimeError):
    """A byte range could not be fetched intact after retries.

    Not an OSError, so tifffile's `except OSError` in is_jfif cannot turn it
    into a silent fallback. That is only a partial defense — tifffile's series
    parser swallows it anyway — so correctness rests on HttpRangeFile.failed,
    the poison flag, not on this type.
    """


class StrictMiss(RuntimeError):
    """A read touched bytes that were not prefetched while network was off."""


class HttpRangeFile(io.RawIOBase):
    """Seekable, thread-safe, read-only file over HTTP Range requests.

    Known bytes live in a sparse in-memory interval store keyed by absolute
    file offset, so there is no offset/alignment arithmetic between a fetched
    range and the bytes served for it. A read of a missing span either
    fetches it (allow_network) or raises StrictMiss. Every response must be a
    206 whose Content-Range and body length match the request exactly.
    """

    def __init__(self, url: str, *, readahead: int = 2 << 10, timeout=(15, 60),
                 retries: int = 5):
        super().__init__()
        self.url = url
        self.name = os.path.basename(urlparse(url).path)
        self.readahead = readahead
        self.timeout = timeout
        self.retries = retries
        self.allow_network = True
        self.miss_phase: str | None = None     # non-strict: on-demand reads count here
        self.failed: BaseException | None = None   # poison flag: any fetch that failed
        self._pos = 0
        self._lock = threading.RLock()
        self._starts: list[int] = []
        self._chunks: dict[int, bytes] = {}
        self._tls = threading.local()
        self._sessions: list[requests.Session] = []
        self._pool: ThreadPoolExecutor | None = None
        self._pool_workers = 0
        self.phase = "open"
        self.stats: dict[str, dict[str, int]] = {}
        self.size = -1
        self._fetch(0, readahead, probe=True)   # doubles as the size probe

    # ---- accounting ------------------------------------------------------
    def _count(self, body: int) -> None:
        with self._lock:
            s = self.stats.setdefault(self.phase, {"requests": 0, "bytes": 0})
            s["requests"] += 1
            s["bytes"] += body

    def totals(self) -> dict[str, int]:
        return {"requests": sum(s["requests"] for s in self.stats.values()),
                "bytes": sum(s["bytes"] for s in self.stats.values())}

    # ---- network -------------------------------------------------------
    def _session(self) -> requests.Session:
        s = getattr(self._tls, "s", None)
        if s is None:
            s = requests.Session()
            s.headers["Accept-Encoding"] = "identity"
            self._tls.s = s
            with self._lock:
                self._sessions.append(s)
        return s

    def _fetch(self, start: int, end: int, probe: bool = False) -> None:
        """GET bytes [start, end) and store them. Refuses anything but an exact 206."""
        if not probe:
            end = min(end, self.size)
        if end <= start:
            return
        last_err: BaseException | None = None
        for attempt in range(self.retries):
            try:
                r = self._session().get(
                    self.url, headers={"Range": f"bytes={start}-{end - 1}"},
                    stream=True, timeout=self.timeout)
                if r.status_code != 206:
                    r.close()     # never let a 200 stream the whole object
                    raise OSError(f"expected 206, got {r.status_code}")
                if r.headers.get("Content-Encoding", "identity") != "identity":
                    r.close()
                    raise OSError("unexpected content-encoding")
                span, total = r.headers["Content-Range"].split(" ", 1)[1].split("/")
                a, b = (int(v) for v in span.split("-"))
                data = r.content
                self._count(len(data))
                if probe:
                    self.size = int(total)
                    end = min(end, self.size)
                if a != start or b != end - 1 or len(data) != end - start:
                    raise OSError(f"short/misaligned range: asked {start}-{end - 1}, "
                                  f"got {a}-{b} len={len(data)}")
                self._insert(start, data)
                return
            except (requests.RequestException, OSError, KeyError, ValueError) as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        err = RangeFetchError(f"{self.name}: range {start}-{end - 1} failed after "
                              f"{self.retries} attempts: {last_err}")
        with self._lock:
            if self.failed is None:
                self.failed = err
        raise err

    def check(self, where: str) -> None:
        """Raise if any fetch failed — including ones a library swallowed."""
        if self.failed is not None:
            raise RangeFetchError(f"{self.name}: a range fetch failed during {where}; "
                                  f"refusing to trust this slide ({self.failed})")

    # ---- interval store -------------------------------------------------
    def _insert(self, start: int, data: bytes) -> None:
        with self._lock:
            for a, b in self._missing(start, start + len(data)):
                i = bisect.bisect_left(self._starts, a)
                self._starts.insert(i, a)
                self._chunks[a] = data[a - start:b - start]

    def _missing(self, a: int, b: int) -> list[tuple[int, int]]:
        """Sub-intervals of [a, b) not covered by stored chunks."""
        gaps = []
        with self._lock:
            i = bisect.bisect_right(self._starts, a) - 1
            cur = a
            if i >= 0:
                s = self._starts[i]
                cur = max(cur, s + len(self._chunks[s]))
            i += 1
            while cur < b and i < len(self._starts):
                s = self._starts[i]
                if s >= b:
                    break
                if s > cur:
                    gaps.append((cur, s))
                cur = max(cur, s + len(self._chunks[s]))
                i += 1
            if cur < b:
                gaps.append((cur, b))
        return gaps

    def _copy(self, a: int, b: int, out: memoryview) -> None:
        with self._lock:
            i = bisect.bisect_right(self._starts, a) - 1
            pos = a
            while pos < b:
                s = self._starts[i]
                chunk = self._chunks[s]
                e = s + len(chunk)
                if not s <= pos < e:
                    raise RuntimeError(f"interval store corrupt at {pos} (chunk {s}-{e})")
                take = min(e, b) - pos
                out[pos - a:pos - a + take] = chunk[pos - s:pos - s + take]
                pos += take
                i += 1

    def stored_bytes(self) -> int:
        return sum(len(c) for c in self._chunks.values())

    # ---- prefetch -------------------------------------------------------
    def prefetch(self, ranges, *, workers: int = 12, max_request: int = 4 << 20) -> int:
        """Fetch many (offset, length) ranges concurrently; returns #requests.

        Exactly-adjacent ranges are coalesced (no byte outside a needed tile is
        fetched); spans larger than max_request are split so a single throttled
        S3 connection never becomes the bottleneck.
        """
        merged: list[list[int]] = []
        for a, b in sorted((o, o + n) for o, n in ranges if n > 0):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        jobs = [(s, min(s + max_request, gb))
                for a, b in merged for ga, gb in self._missing(a, b)
                for s in range(ga, gb, max_request)]
        if not jobs:
            return 0
        jobs.sort(key=lambda j: j[0] - j[1])          # biggest first: better tail latency
        if self._pool is None or self._pool_workers != workers:
            if self._pool is not None:
                self._pool.shutdown()
            self._pool = ThreadPoolExecutor(max_workers=workers)
            self._pool_workers = workers
        list(self._pool.map(lambda j: self._fetch(*j), jobs))
        return len(jobs)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        for s in self._sessions:
            s.close()
        self._sessions.clear()
        self._chunks.clear()
        self._starts.clear()
        super().close()

    # ---- RawIOBase ------------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        with self._lock:
            if whence == 0:
                self._pos = offset
            elif whence == 1:
                self._pos += offset
            elif whence == 2:
                self._pos = self.size + offset
            else:
                raise ValueError(whence)
            return self._pos

    def readall(self) -> bytes:
        raise StrictMiss("unbounded read refused (would download the whole slide)")

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            return self.readall()
        buf = bytearray(n)
        got = self.readinto(buf)
        return bytes(buf[:got])

    def readinto(self, b) -> int:
        out = memoryview(b).cast("B")
        with self._lock:
            a = self._pos
            end = min(a + len(out), self.size)
            if a >= end:
                return 0
            gaps = self._missing(a, end)
            if gaps:
                if not self.allow_network:
                    if self.miss_phase is None:
                        raise StrictMiss(f"strict: read {a}+{end - a} not prefetched "
                                         f"(missing {gaps[:3]})")
                    saved, self.phase = self.phase, self.miss_phase
                    try:
                        for ga, gb in gaps:
                            self._fetch(ga, gb)
                    finally:
                        self.phase = saved
                else:
                    for ga, gb in gaps:
                        if gb == end:   # read-ahead only past the end of the request
                            nxt = bisect.bisect_left(self._starts, gb)
                            cap = self._starts[nxt] if nxt < len(self._starts) else self.size
                            gb = min(max(gb, ga + self.readahead), cap)
                        self._fetch(ga, gb)
            self._copy(a, end, out)
            self._pos = end
            return end - a


def _region_tiles(page, regions) -> list[int]:
    """TIFF tile indices of `page` overlapped by level-space regions [x0,x1)x[y0,y1)."""
    if not page.is_tiled:
        raise NotImplementedError("level is not tiled")
    if page.planarconfig != 1:
        raise NotImplementedError("planar-separate TIFF not supported")
    tw, tl = page.tilewidth, page.tilelength
    W, H = page.imagewidth, page.imagelength
    ntx = math.ceil(W / tw)
    need = set()
    for x0, y0, x1, y1 in regions:
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        for ty in range(y0 // tl, (y1 - 1) // tl + 1):
            for tx in range(x0 // tw, (x1 - 1) // tw + 1):
                need.add(ty * ntx + tx)
    return sorted(need)


def _level_page(slide, level: int):
    if slide.properties.get("tiffslide.series-composition"):
        raise NotImplementedError("composited series not supported")
    sidx = slide.properties.get("tiffslide.series-index", 0) or 0
    return slide.ts_tifffile.series[sidx].levels[level].keyframe


def _tile_ranges(page, idx) -> list[tuple[int, int]]:
    offs, cnts = page.dataoffsets, page.databytecounts
    return [(int(offs[i]), int(cnts[i])) for i in idx if offs[i] > 0 and cnts[i] > 0]


def iter_tiles_remote(url: str, *, target_mpp: float, patch: int, workers: int = 12,
                      strict: bool = False, report: dict | None = None):
    """Yield (coords, level, scale) first, then one float32 tile per coord.

    Byte-identical to camelyon_lib's local path (open_slide -> pick_level ->
    tissue_tiles -> read_patches), because it IS that path on a remote-backed
    file. target_mpp and patch are required: a silent geometry default is how
    two runs end up comparing different tiles.

    strict=False (production): a read the prefetch plan missed is fetched on
    demand, counted as a plan miss, and output stays identical. strict=True
    (verification): such a read raises StrictMiss. Either way a failed fetch
    raises RangeFetchError and poisons the slide.
    """
    import tiffslide

    rep = report if report is not None else {}
    rf = HttpRangeFile(url)
    slide = None
    try:
        rep["file_size"] = rf.size
        rf.phase = "header"
        slide = tiffslide.TiffSlide(rf)
        _ = dict(slide.properties), slide.level_dimensions, slide.level_downsamples
        _ = slide.zarr_group
        level, scale = pick_level(slide, target_mpp=target_mpp)
        thumb_level = len(slide.level_downsamples) - 1
        tpage, lpage = _level_page(slide, thumb_level), _level_page(slide, level)
        # Decisions that peek at tile bytes (is_jfif reads the first tile) are
        # made NOW, from real bytes, with the network on — then the poison flag
        # proves no fetch failure was swallowed while making them.
        for pg in (tpage, lpage):
            _ = pg.is_jfif, pg.decode
        ntx = math.ceil(lpage.imagewidth / lpage.tilewidth)
        nty = math.ceil(lpage.imagelength / lpage.tilelength)
        if len(lpage.dataoffsets) != ntx * nty:
            raise NotImplementedError(f"unexpected tile grid: {len(lpage.dataoffsets)} "
                                      f"offsets for {ntx}x{nty} tiles")
        rf.check("header parsing")

        # from here on, network only via explicit prefetch (or counted misses)
        rf.allow_network = False
        rf.miss_phase = None if strict else "miss"

        rf.phase = "thumb"
        tw, th = slide.level_dimensions[thumb_level]
        rf.prefetch(_tile_ranges(tpage, _region_tiles(tpage, [(0, 0, tw, th)])),
                    workers=workers)
        coords = tissue_tiles(slide, level, patch=patch)

        # prefetch exactly the level tiles read_patches will touch (see module doc)
        rf.phase = "level"
        ds = slide.level_downsamples[level]
        side = patch if abs(scale - 1) < 0.02 else int(round(patch / scale))
        regions = []
        for x, y in coords:
            rx0, ry0 = int(int(x * ds) / ds), int(int(y * ds) / ds)   # read_region transform
            regions.append((rx0, ry0, rx0 + side, ry0 + side))
        rep["level_requests"] = rf.prefetch(
            _tile_ranges(lpage, _region_tiles(lpage, regions)), workers=workers)
        rf.check("prefetch")

        yield coords, level, scale
        rf.phase = "decode"
        for tile in read_patches(slide, level, coords, scale, patch=patch):
            yield tile
        rf.check("decode")
        tot = rf.totals()
        rep["bytes_fetched"] = tot["bytes"]
        rep["requests"] = tot["requests"]
        rep["fraction_of_file"] = tot["bytes"] / rf.size if rf.size > 0 else None
        rep["plan_misses"] = rf.stats.get("miss", {}).get("requests", 0)
        rep["peak_memory_bytes"] = rf.stored_bytes()
    finally:
        if slide is not None:
            slide.close()      # tifffile does not close a caller-provided stream
        rf.close()             # frees the in-memory tile bytes and sockets
