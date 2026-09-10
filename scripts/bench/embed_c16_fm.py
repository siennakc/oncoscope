"""Embed CAMELYON16 slides with a pathology foundation model: one bag per slide.

Dean Tessone's feedback (2026-09-02): ImageNet/PCam encoders leave performance
on the table — "Virchow2 based models show substantially better performance."
This script swaps the encoder: every tissue tile becomes an FM embedding, and
the per-slide bag (N x D fp16 + tile coords) feeds ABMIL downstream
(src/oncoscope/models/abmil.py, scripts/bench/train_abmil.py).

Streaming discipline matches infer_c16_slide.py: fetch one slide, tile,
embed, save the bag, delete the slide. Resumable — cached bags are skipped.
Bags are written atomically and downloads are size+magic validated, so an
interrupted run never leaves a corrupt file that a re-run would trust; a
slide yielding zero tissue tiles is recorded in _failed.json, not cached.

Geometry: FMs are trained on 224px tiles at ~0.5 um/px (20x), so the default
here is --mpp 0.5 / 224px — NOT the 0.972/96px PCam geometry (pass
--mpp 0.972 for the 4x-cheaper apples-to-apples frame). Bags are written
under a model+mpp key so geometries can't silently mix.

Model registry (all via HF hub; gated ones need `hf auth login` after access
is granted on the model page):

  hibou_b     open   ViT-B/14, 768-d   (transformers, trust_remote_code —
                                        executes model code from histai's repo)
  phikon      open   ViT-B/16, 768-d   (transformers, standard ViT class, no
                                        remote code, ~350MB — safest smoke test)
  phikon_v2   open   ViT-L/16, 1024-d  (transformers, standard Dinov2 class,
                                        no remote code)
  h_optimus_0 open   ViT-g/14, 1536-d  (timm; 1.1B params — GPU machine only)
  virchow2    GATED  ViT-H/14, 2560-d  (timm; Dean's reference model)
  uni         GATED  ViT-L/16, 1024-d  (timm; Mahmood lab)

Usage:
  python scripts/bench/embed_c16_fm.py --model hibou_b --list test_001,test_002
  python scripts/bench/embed_c16_fm.py --model virchow2 --list-file train_slides.txt
  python scripts/bench/embed_c16_fm.py --smoke        # random tiles, no download of slides
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/bench")
import torch  # noqa: E402

from camelyon_lib import (  # noqa: E402
    BUCKET, fetch, open_slide, pick_level, read_patches, tissue_tiles,
)

FM_PATCH = 224
FM_MPP = 0.5

REGISTRY = {
    "hibou_b": dict(kind="transformers", repo="histai/hibou-b", dim=768,
                    gated=False, remote_code=True, pool="pooler"),
    "phikon": dict(kind="transformers", repo="owkin/phikon", dim=768,
                   gated=False, remote_code=False, pool="cls"),
    "phikon_v2": dict(kind="transformers", repo="owkin/phikon-v2", dim=1024,
                      gated=False, remote_code=False, pool="cls"),
    # kaiko-ai Midnight-12k: ungated + MIT, C16-small 0.869 (arXiv:2504.05186
    # Table 2) — the strongest no-gate option. Card recipe: concat CLS with the
    # mean of the patch tokens (3072-d). Smoke-test before a long run.
    "midnight": dict(kind="transformers", repo="kaiko-ai/midnight", dim=3072,
                     gated=False, remote_code=False, pool="cls_mean"),
    "h_optimus_0": dict(kind="timm", repo="hf-hub:bioptimus/H-optimus-0", dim=1536,
                        gated=False,
                        timm_kwargs=dict(init_values=1e-5, dynamic_img_size=False)),
    "virchow2": dict(kind="timm", repo="hf-hub:paige-ai/Virchow2", dim=2560,
                     gated=True, pool="virchow"),
    "uni": dict(kind="timm", repo="hf-hub:MahmoodLab/UNI", dim=1024, gated=True,
                timm_kwargs=dict(init_values=1e-5, dynamic_img_size=True)),
    # UNI2-h: top of the eva table (C16-small 0.873) but institutional-email
    # gate + pickle-only weights — exact create_model kwargs from the card.
    "uni2_h": dict(kind="timm", repo="hf-hub:MahmoodLab/UNI2-h", dim=1536,
                   gated=True,
                   timm_kwargs=dict(img_size=224, patch_size=14, depth=24,
                                    num_heads=24, embed_dim=1536,
                                    mlp_ratio=2.66667 * 2, no_embed_class=True,
                                    reg_tokens=8, init_values=1e-5,
                                    dynamic_img_size=True, swiglu=True)),
    # Generic (non-pathology) ImageNet ResNet-50, GAP 2048-d. The control the
    # FM literature reports against: run at the SAME patch/mpp as the FM so the
    # ONLY difference is the encoder. Weaker than pcam_resnet (which is
    # fine-tuned on lymph-node patches), so it flatters the FM — say so in any
    # writeup, and prefer pcam_resnet when its checkpoint is available.
    "imagenet_resnet": dict(kind="imagenet", dim=2048, gated=False),
    # The CURRENT tile encoder's features (PCam-trained ResNet-50, GAP 2048-d,
    # raw [0,1] input, 96px @ 0.972 mpp — the geometry it was trained on).
    # Purpose: the ResNet+ABMIL cell of the 2x2, isolating what the FM buys
    # vs what ABMIL buys on the SAME features as the 0.827 baseline.
    "pcam_resnet": dict(kind="pcam", weights="runs/pcam/best_model.pt", dim=2048,
                        gated=False, patch=96, default_mpp=0.972),
}


def load_fm(key: str, device: torch.device):
    """Returns (embed_fn: (B,3,H,W) float [0,1] -> (B,D) fp32 numpy, dim)."""
    spec = REGISTRY[key]
    if spec["kind"] == "pcam":
        import torchvision
        if not Path(spec["weights"]).exists():
            raise SystemExit(
                f"[fm] {key} needs {spec['weights']} (the PCam-trained "
                "ResNet-50) — it lives on the training machine; run this "
                "model there, or copy the checkpoint over first")
        net = torchvision.models.resnet50(weights=None)
        net.fc = torch.nn.Identity()
        state = torch.load(spec["weights"], map_location="cpu",
                           weights_only=True)["model"]
        # strict: a key mismatch must fail here, not leave a random ResNet
        # silently standing in as the "PCam" encoder for a whole run
        net.load_state_dict({k: v for k, v in state.items()
                             if not k.startswith("fc.")}, strict=True)
        net = net.eval().to(device)

        @torch.no_grad()
        def embed(x):  # trained on raw [0,1] RGB — no normalization
            return net(x.to(device)).float().cpu().numpy()

        return embed, spec["dim"]

    if spec["kind"] == "imagenet":
        import torchvision
        net = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = torch.nn.Identity()
        net = net.eval().to(device)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(device)
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(device)

        @torch.no_grad()
        def embed(x):
            return net((x.to(device) - mean) / std).float().cpu().numpy()

        return embed, spec["dim"]

    if spec["kind"] == "timm":
        import timm
        from timm.data import resolve_model_data_config
        kwargs = dict(spec.get("timm_kwargs", {}))
        if kwargs.pop("swiglu", False) or spec.get("pool") == "virchow":
            from timm.layers import SwiGLUPacked
            kwargs.update(mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        try:
            model = timm.create_model(spec["repo"], pretrained=True, num_classes=0,
                                      **kwargs)
        except Exception as e:  # gated repos raise on the hub call
            if spec["gated"]:
                raise SystemExit(
                    f"[fm] {key} is GATED on Hugging Face. Request access at "
                    f"https://huggingface.co/{spec['repo'].removeprefix('hf-hub:')} "
                    f"then `hf auth login`. Original error: {e}")
            raise
        cfg = resolve_model_data_config(model)
        mean = torch.tensor(cfg["mean"])[:, None, None].to(device)
        std = torch.tensor(cfg["std"])[:, None, None].to(device)
        model = model.eval().to(device)

        @torch.no_grad()
        def embed(x):
            x = (x.to(device) - mean) / std
            if spec.get("pool") == "virchow":
                # Virchow2 convention: concat CLS with mean of patch tokens
                # (tokens 1..4 are registers, skipped)
                tokens = model.forward_features(x)
                out = torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=-1)
            else:
                out = model(x)  # num_classes=0 -> pooled embedding
            return out.float().cpu().numpy()

        return embed, spec["dim"]

    from transformers import AutoImageProcessor, AutoModel
    if spec["remote_code"]:
        print(f"[fm] NOTE: {key} loads model code from the "
              f"{spec['repo']} repo (trust_remote_code=True)", flush=True)
    proc = AutoImageProcessor.from_pretrained(spec["repo"],
                                              trust_remote_code=spec["remote_code"])
    model = AutoModel.from_pretrained(spec["repo"],
                                      trust_remote_code=spec["remote_code"])
    model = model.eval().to(device)
    mean = torch.tensor(proc.image_mean)[:, None, None].to(device)
    std = torch.tensor(proc.image_std)[:, None, None].to(device)

    @torch.no_grad()
    def embed(x):
        x = (x.to(device) - mean) / std
        out = model(pixel_values=x)
        if spec.get("pool") == "pooler" and getattr(out, "pooler_output", None) is not None:
            emb = out.pooler_output
        elif spec.get("pool") == "cls_mean":  # midnight: concat CLS + patch mean
            h = out.last_hidden_state
            emb = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)
        else:  # canonical for phikon family: raw CLS token
            emb = out.last_hidden_state[:, 0]
        return emb.float().cpu().numpy()

    return embed, spec["dim"]


def smoke(key: str, device: torch.device) -> None:
    """Load the FM and embed random tiles — proves wiring without any slide."""
    embed, dim = load_fm(key, device)
    patch = REGISTRY[key].get("patch", FM_PATCH)
    x = torch.rand(4, 3, patch, patch)
    t0 = time.time()
    out = embed(x)
    assert out.shape == (4, dim), f"expected (4,{dim}), got {out.shape}"
    assert np.isfinite(out).all()
    # identical tiles must embed identically (eval mode, no dropout)
    twice = embed(torch.cat([x[:1], x[:1]]))
    assert np.allclose(twice[0], twice[1], atol=1e-4)
    print(f"[fm] SMOKE OK {key}: dim={dim}, 4 tiles in {time.time()-t0:.1f}s, "
          f"norm mean={np.linalg.norm(out, axis=1).mean():.1f}", flush=True)


def save_bag(out: Path, **arrays) -> None:
    """Write the .npz atomically: a crash mid-write must not leave a file
    that a resumed run would skip as 'cached'."""
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, out)


def record_failure(out_dir: Path, name: str, reason: str) -> Path:
    failed = out_dir / "_failed.json"
    rec = json.loads(failed.read_text()) if failed.exists() else {}
    rec[name] = reason
    failed.write_text(json.dumps(rec, indent=1))
    return failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phikon",
                    help="encoder, or a comma-separated list to embed every "
                         "slide with several encoders in ONE download pass "
                         f"(choices: {', '.join(sorted(REGISTRY))})")
    ap.add_argument("--list", default="")
    ap.add_argument("--list-file", default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mpp", type=float, default=FM_MPP,
                    help="target um/px (0.5 = FM-native 20x; 0.972 = PCam geometry)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent byte-range connections per slide; S3 throttles "
                         "a single connection well below the link (2 vs 17 MB/s here). "
                         "1 restores plain resumable curl")
    ap.add_argument("--device", default=None,
                    help="cuda / mps / cpu (auto-detected; CPU must be explicit)")
    ap.add_argument("--smoke", action="store_true",
                    help="load model, embed random tiles, exit")
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[fm] device: {dev}", flush=True)
    models = [m for m in args.model.split(",") if m]
    unknown = [m for m in models if m not in REGISTRY]
    if unknown:
        raise SystemExit(f"[fm] unknown model(s) {unknown}; choices: {sorted(REGISTRY)}")
    if args.smoke:
        for m in models:
            smoke(m, dev)
        return
    if dev.type == "cpu" and args.device != "cpu":
        raise SystemExit("[fm] REFUSED: no CUDA/MPS device found — a CPU embed is "
                         "~10x slower and would silently turn an overnight job "
                         "into a week; pass --device cpu to insist")

    names = ([n for n in args.list.split(",") if n] if args.list
             else [l.strip() for l in open(args.list_file) if l.strip()])
    if not names:
        raise SystemExit("no slides: pass --list or --list-file (or --smoke)")

    # Every encoder in one pass must see IDENTICAL tiles — that is the whole
    # point of the comparison, and it is also what makes the extra encoder
    # nearly free on a download-bound job. So they must agree on tile size.
    patches = {m: REGISTRY[m].get("patch", FM_PATCH) for m in models}
    if len(set(patches.values())) > 1:
        raise SystemExit(f"[fm] REFUSED: encoders disagree on tile size {patches} — "
                         "one pass must produce identical tiles for every encoder; "
                         "run the differing one separately")
    patch = next(iter(patches.values()))
    if len(models) == 1 and args.mpp == FM_MPP and "default_mpp" in REGISTRY[models[0]]:
        args.mpp = REGISTRY[models[0]]["default_mpp"]
        print(f"[fm] {models[0]}: using its native geometry "
              f"{patch}px @ {args.mpp} um/px", flush=True)

    embedders = {}
    for m in models:
        fn, d = load_fm(m, dev)
        run_key = f"{m}_mpp{args.mpp:g}"
        od = Path("runs/c16/embeds") / run_key
        od.mkdir(parents=True, exist_ok=True)
        embedders[m] = (fn, d, od)
        print(f"[fm] {m}: dim={d} -> {od}", flush=True)
    print(f"[fm] geometry: {patch}px @ {args.mpp} um/px, {len(models)} encoder(s) "
          "per downloaded slide", flush=True)
    scratch = Path("runs/c16/slides")
    scratch.mkdir(parents=True, exist_ok=True)

    def pending(nm):
        """Encoders that still owe a bag for this slide."""
        return [m for m in models if not (embedders[m][2] / f"{nm}.npz").exists()]

    # No background prefetch: fetch() now saturates the link with concurrent
    # byte ranges, so a second stream would only contend for the same ceiling
    # (and its partial file would be discarded by the range download anyway).
    # Peak disk stays exactly one slide.
    for idx, name in enumerate(names):
        todo = pending(name)
        if not todo:
            print(f"[fm] {name}: cached", flush=True)
            continue
        t0 = time.time()
        slide_path = scratch / f"{name}.tif"
        fetch(f"images/{name}.tif", slide_path, workers=args.workers)
        t_dl = time.time() - t0
        slide = open_slide(slide_path)
        level, scale = pick_level(slide, target_mpp=args.mpp)
        coords = tissue_tiles(slide, level, patch=patch)
        embs = {m: [] for m in todo}
        batch = []
        for tile in read_patches(slide, level, coords, scale, patch=patch):
            batch.append(torch.from_numpy(tile).permute(2, 0, 1))
            if len(batch) == args.batch:
                x = torch.stack(batch)
                for m in todo:
                    embs[m].append(embedders[m][0](x))
                batch = []
        if batch:
            x = torch.stack(batch)
            for m in todo:
                embs[m].append(embedders[m][0](x))
        slide.close()
        if not args.keep:
            slide_path.unlink(missing_ok=True)
        if not any(embs.values()):
            for m in todo:
                failed = record_failure(embedders[m][2], name,
                                        "0 tissue tiles — inspect the thumbnail; "
                                        "the tissue mask found nothing")
            print(f"[fm] {name}: FAILED, 0 tissue tiles — not cached, recorded in "
                  f"{failed}", flush=True)
            continue
        n_tiles, dims = 0, []
        for m in todo:
            fn, dim, od = embedders[m]
            bag = np.concatenate(embs[m])
            n_tiles, _ = bag.shape[0], dims.append(f"{m}:{dim}d")
            if bag.shape[0] < 200:
                print(f"[fm] WARNING {name}: only {bag.shape[0]} tissue tiles — "
                      "check the thumbnail before trusting this bag", flush=True)
            save_bag(
                od / f"{name}.npz", embeddings=bag.astype(np.float16),
                coords=np.array(coords, np.int32).reshape(-1, 2),
                level=level, scale=scale, patch=patch, target_mpp=args.mpp,
                model=m, repo=REGISTRY[m].get("repo", m))
            (od / f"{name}.json").write_text(json.dumps(
                {"slide": name, "model": f"{m}_mpp{args.mpp:g}",
                 "n_tiles": int(bag.shape[0]), "dim": dim, "level": level,
                 "dl_s": round(t_dl, 1), "total_s": round(time.time() - t0, 1)}))
        print(f"[fm] {name}: {n_tiles} tiles x [{', '.join(dims)}] "
              f"({time.time() - t0:.0f}s, dl {t_dl:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
