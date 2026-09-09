"""CAMELYON16 streaming toolkit: fetch one slide, tile tissue at 10x, score, delete.

Slides are BigTIFF pyramids on public S3 (CC0). Nothing here stores more than
one slide at a time (peak disk < 5 GB/worker against a 340 GB corpus).
Patch geometry matches PCam exactly: 96x96 at ~0.97 um/px, so the PCam-trained
classifier sees its native distribution.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path
import numpy as np

BUCKET = "https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON16"
TARGET_MPP = 0.972
PATCH = 96
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def stratified_split(names, seed: int = 0, val_frac: float = 0.2):
    """Deterministic stratified train/val split of C16 TRAIN slide names.

    Single source of truth: train_abmil.py and build_c16_manifests.py both
    call this, so the registered protocol and the committed manifests cannot
    disagree.
    """
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for prefix in ("normal", "tumor"):
        group = sorted(n for n in names if n.startswith(prefix))
        perm = rng.permutation(len(group))
        cut = max(1, int(round(len(group) * val_frac)))
        va += [group[i] for i in perm[:cut]]
        tr += [group[i] for i in perm[cut:]]
    return sorted(tr), sorted(va)


def remote_size(url: str) -> int | None:
    """Content-Length from a HEAD request (None if unavailable/offline)."""
    out = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True)
    sizes = re.findall(r"(?im)^content-length:\s*(\d+)", out.stdout or "")
    return int(sizes[-1]) if sizes else None


def looks_complete(path: Path, size: int | None) -> bool:
    """Trust a slide file only if it is a TIFF and (when known) full-size.

    curl without --fail writes an S3 error body as the .tif, and an
    interrupted download leaves a truncated pyramid that tiffslide reads
    partially — both must count as absent, never as cached.
    """
    if not path.exists():
        return False
    if path.suffix.lower() in (".tif", ".tiff"):
        with path.open("rb") as fh:
            if fh.read(4) not in TIFF_MAGIC:
                return False
    return size is None or path.stat().st_size == size


def fetch(rel: str, dest: Path) -> Path:
    """Resumable download to dest.part, validated, then renamed into place.

    dest only ever holds a complete, validated file; a stale/corrupt dest
    (from an older run) is replaced.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET}/{rel}"
    size = remote_size(url)
    if looks_complete(dest, size):
        return dest
    if dest.exists():
        print(f"[fetch] {dest.name}: stale or corrupt, re-downloading", flush=True)
        dest.unlink()
    part = dest.with_name(dest.name + ".part")
    if not (size is not None and part.exists() and part.stat().st_size >= size):
        subprocess.run(["curl", "-fsSL", "--retry", "5", "-C", "-", "-o", str(part), url],
                       check=True)
    if not looks_complete(part, size):
        got = part.stat().st_size if part.exists() else 0
        part.unlink(missing_ok=True)
        raise RuntimeError(f"[fetch] {rel}: download invalid ({got} bytes, expected "
                           f"{size}) — not a complete TIFF; removed, re-run to retry")
    part.replace(dest)
    return dest


def open_slide(path: Path):
    import tiffslide
    return tiffslide.TiffSlide(str(path))


def pick_level(slide, target_mpp: float = TARGET_MPP) -> tuple[int, float]:
    """Level whose mpp is closest to target_mpp, plus its scale vs target."""
    mpp0 = float(slide.properties.get("tiffslide.mpp-x") or 0.243)
    best, best_d = 0, 1e9
    for lvl, ds in enumerate(slide.level_downsamples):
        d = abs(mpp0 * ds - target_mpp)
        if d < best_d:
            best, best_d = lvl, d
    return best, (mpp0 * slide.level_downsamples[best]) / target_mpp


def tissue_tiles(slide, level: int, stride: int | None = None, patch: int = PATCH):
    """Grid coords (level space) whose thumbnail cell looks like tissue (HSV).

    ``patch`` sets the tile side in level pixels (96 for the PCam classifier,
    224 for pathology foundation models); stride defaults to non-overlapping.
    """
    stride = patch if stride is None else stride
    thumb_level = len(slide.level_downsamples) - 1
    tw, th = slide.level_dimensions[thumb_level]
    thumb = np.asarray(slide.read_region((0, 0), thumb_level, (tw, th)).convert("RGB"),
                       np.float32) / 255.0
    mx, mn = thumb.max(2), thumb.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    mask = (sat > 0.07) & (mx > 0.1) & (mn < 0.95)
    W, H = slide.level_dimensions[level]
    fx, fy = tw / W, th / H
    coords = []
    for y in range(0, H - patch + 1, stride):
        ty0, ty1 = int(y * fy), max(int((y + patch) * fy), int(y * fy) + 1)
        row = mask[ty0:ty1]
        for x in range(0, W - patch + 1, stride):
            tx0, tx1 = int(x * fx), max(int((x + patch) * fx), int(x * fx) + 1)
            if row[:, tx0:tx1].mean() > 0.25:
                coords.append((x, y))
    return coords


def read_patches(slide, level: int, coords, scale: float, patch: int = PATCH):
    """Yield float32 [0,1] RGB patch x patch tiles (resampled if level mpp != target)."""
    from PIL import Image
    ds = slide.level_downsamples[level]
    # scale = level_mpp / target_mpp: a finer level (scale < 1) needs MORE
    # level pixels to cover the target field of view before downsampling
    side = patch if abs(scale - 1) < 0.02 else int(round(patch / scale))
    for x, y in coords:
        img = slide.read_region((int(x * ds), int(y * ds)), level, (side, side)).convert("RGB")
        if side != patch:
            img = img.resize((patch, patch), Image.BILINEAR)
        yield np.asarray(img, np.float32) / 255.0
