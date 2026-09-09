# Training-machine handoff — FM + ABMIL round (Dean's feedback, made rigorous)

You are an AI agent working in the `oncoscope` checkout on the training machine
(Apple Silicon, MPS, `data/raw/` populated, `.venv` with torch). This document
is your work order. It was written by the agent working with Sienna on her
machine on 2026-09-07; everything referenced here is committed on `main`.
The previous work order (patch stage + A/B rematches) is DONE — its results
are in `TRAINING_HISTORY.md` Part 8.

Context: Dean Tessone (CSI-Cancer) reviewed the project and named two upgrades
for the CAMELYON16 lane — a pathology foundation model instead of the
ImageNet/PCam encoder, and ABMIL instead of top5_mean. Both are now
implemented; this round produces the measured answer. The research and
fact-checked model comparison live in the "Oncoscope FM Field Guide" artifact;
the short version is inline below.

## Prime directives (read before any command)

1. **Refusals are features.** Scripts enforce provenance, seals, and the query
   budget in code. If a script refuses, it is working — diagnose, never
   bypass. In particular `--exclude` exists to *record* a deliberate dropped
   slide in the protocol; it is not a way to silence a missing-bag refusal you
   have not understood.
2. **The C16 official test is ONE query this round, and the code now enforces
   it globally.** `data/manifests/c16_abmil_v1/official_queries.jsonl` is the
   ledger: `--official-test` refuses once `query_budget` (1, in
   `dataset.json`) is spent, no matter which encoder, seed, or output
   directory you run from. Another seed is NOT another shot. Every other cell
   of the 2×2 below is reported on the 54-slide val split only. A second query
   is a governance decision — you raise the budget in `dataset.json`
   deliberately, in a commit, with the reason — never a rerun.
3. **Sealed sets from the mammography lane remain off-limits** (sealed_test_v1;
   MIAS budget 20, 2 spent). Unchanged.
4. **Commit conventions:** author Sienna Chen, no AI co-author trailers,
   result JSONs/CARDs committed, weights to GitHub releases, raw data and
   embedding caches never in git. Pull before you start; rebase, never
   force-push. `runs/` is gitignored — the committable artifacts are mirrored
   to `results/c16_abmil/<run_key>/` for you.
5. **Geometry is part of every claim.** Round 1 runs at `--mpp 0.972`
   (the PCam geometry — 4× cheaper than the FM-native 0.5, and the honest
   apples-to-apples frame against the 0.827 baseline). Every number you write
   down states its mpp. Do not mix geometries inside a comparison.

## Step 0 — Sync and verify (10 min)

```sh
git pull --rebase
.venv/bin/pip install -e '.[dev,fm]' --quiet
.venv/bin/python -m pytest -q
```

- Expect **100 passed**. The `fm` extra is new (torch/timm/transformers/
  tiffslide) — without it the embedder cannot import.
- New since the patch stage: `src/oncoscope/models/abmil.py` (gated ABMIL),
  `scripts/bench/embed_c16_fm.py` (FM bag embedder + model registry),
  `scripts/bench/train_abmil.py` (manifest-driven trainer, protocol
  pre-registration, ledger-enforced official test, attention export/render),
  `scripts/build_c16_manifests.py` + `data/manifests/c16_abmil_v1/` (sealed
  manifests: 216 train / 54 val / 129 test), `tests/test_camelyon_lib.py`,
  and `camelyon_lib` grew `stratified_split`, per-call `patch`/`target_mpp`,
  and a validating `fetch`.
- Smoke the wiring before anything long:
  `.venv/bin/python scripts/bench/embed_c16_fm.py --model phikon --smoke`
  and `--model pcam_resnet --smoke` (the latter needs `runs/pcam/best_model.pt`,
  which lives on this machine only). Expect `SMOKE OK` with dims 768 and 2048.

## Step 0.5 — HuggingFace auth (Sienna's step, not yours)

Gated models (`virchow2`, `uni`, `uni2_h`) need Sienna's HF token on this
machine: she runs `.venv/bin/hf auth login` herself and pastes her token.
Never handle the token for her. Access on the hub side is already granted
(MahmoodLab + paige-ai, approved 2026-09-03). Phikon and `pcam_resnet` need no
auth — do not block on this step.

## Step 1 — Round-1 embeddings (the long job; start it first)

Slide lists come from the sealed manifests:

```sh
cd data/manifests/c16_abmil_v1
tail -n +2 train.csv | cut -d, -f1 > /tmp/c16_train.txt        # 216
tail -n +2 val.csv   | cut -d, -f1 > /tmp/c16_val.txt          #  54
tail -n +2 test_SEALED.csv | cut -d, -f1 > /tmp/c16_test.txt   # 129
cd ../../..

# 1a. Phikon bags, train+val (resumable; ~12-25 h embed + download —
#     run in a persistent session, re-run on interruption, it skips cached):
.venv/bin/python scripts/bench/embed_c16_fm.py --model phikon --mpp 0.972 \
    --list-file /tmp/c16_train.txt
.venv/bin/python scripts/bench/embed_c16_fm.py --model phikon --mpp 0.972 \
    --list-file /tmp/c16_val.txt

# 1b. pcam_resnet bags, train+val (fast — ResNet-50 at ~100+ t/s, 96px;
#     it auto-selects its native 96px/0.972 geometry, same run_key suffix):
.venv/bin/python scripts/bench/embed_c16_fm.py --model pcam_resnet \
    --list-file /tmp/c16_train.txt
.venv/bin/python scripts/bench/embed_c16_fm.py --model pcam_resnet \
    --list-file /tmp/c16_val.txt
```

Output lands in `runs/c16/embeds/phikon_mpp0.972/` and
`runs/c16/embeds/pcam_resnet_mpp0.972/` — those exact paths are what `--embeds`
wants in Step 2.

Resume semantics (all enforced in code, so trust them): downloads are
size- and TIFF-magic-validated before use, bags are written atomically, and a
slide that yields zero tissue tiles is recorded in `_failed.json` rather than
cached as a good empty bag. Re-running after any interruption is safe.

Sanity rails: ~270 `.npz` bags per model; tiles/slide roughly 2k–15k at
0.972 mpp (the script warns below 200 tiles — look at that slide's thumbnail
before trusting its bag); `dim` 768 (phikon) / 2048 (pcam_resnet). Disk:
phikon bags ~5 GB fp16, pcam_resnet ~13 GB. If `_failed.json` appears,
investigate before Step 2 — a missing bag will (correctly) refuse to train.

## Step 2 — The 2×2 on validation (minutes per cell, no test contact)

The train/val split is the committed manifest, hash-checked on every run;
`--seed` changes model init and shuffling ONLY, so three seeds give a real
seed-variance number on one fixed val set.

```sh
# FM + ABMIL (three seeds — C16 is small; report val mean ± std):
for s in 0 1 2; do
  .venv/bin/python scripts/bench/train_abmil.py \
      --embeds runs/c16/embeds/phikon_mpp0.972 --seed $s
done

# ResNet + ABMIL (isolates the aggregator's contribution):
for s in 0 1 2; do
  .venv/bin/python scripts/bench/train_abmil.py \
      --embeds runs/c16/embeds/pcam_resnet_mpp0.972 --seed $s
done
```

The 2×2 this produces (all val-side except the existing baseline):

|                       | top5_mean            | ABMIL                     |
|-----------------------|----------------------|---------------------------|
| PCam-ResNet features  | 0.827 (official, old)| val mean±std (this step)  |
| Phikon features       | —                    | val mean±std (this step)  |

(The FM+top5_mean cell needs a tile-level probe on FM features — optional;
skip unless the two ABMIL cells leave the attribution genuinely ambiguous.)

Sanity rails: phikon+ABMIL val AUROC should land well above the 0.827-era
regime (the ACMIL literature says ImageNet→SSL features move ABMIL ~0.79→
~0.94; at 0.972 mpp expect something between). If pcam_resnet+ABMIL ≈
phikon+ABMIL, the FM bought little and that is itself the finding — record
it either way. Note the registered model is the LAST epoch of the cosine
schedule, not an epoch-selected best (`model_selection` in protocol.json says
so): val AUROC is therefore an honest held-out number, not a max over epochs.

## Step 3 — Pre-register, then the ONE official test shot

1. Pick the headline config = best val AUROC among the seeds/encoders above.
   **Write down its encoder AND its seed** — you need both in step 4.
2. Its `protocol.json` (already mirrored to
   `results/c16_abmil/<run_key>/protocol.json`) is the pre-registration —
   commit it, and note the chosen run_key plus the val table in the CARD draft
   BEFORE embedding a single test slide.
3. Embed the 129 test slides with the chosen encoder only:
   ```sh
   .venv/bin/python scripts/bench/embed_c16_fm.py --model <chosen> \
       --mpp 0.972 --list-file /tmp/c16_test.txt
   ```
4. One shot. **Pass the winning seed explicitly** — omitting `--seed` means
   seed 0, which would score a different model than the one you pre-registered
   and spend the budget doing it:
   ```sh
   .venv/bin/python scripts/bench/train_abmil.py \
       --embeds runs/c16/embeds/<chosen>_mpp0.972 --seed <winning seed> \
       --official-test
   ```
   The ledger is charged before scoring, so a crash mid-eval still counts as
   the shot — that is deliberate. The aggregate result is mirrored to
   `results/c16_abmil/<run_key>/official_test.json`; per-slide test scores stay
   in `runs/` on purpose (hand-tuning against them is how a sealed set dies).
5. Render 3–5 attention overlays, tumor and normal slides both
   (`--render test_XXX --seed <winning seed>`) — the interpretability artifact
   Dean contrasted with top5_mean. PNGs land in
   `results/c16_abmil/<run_key>/attention/`; commit them with the CARD.

## Step 4 — Write it up

`results/c16_abmil/CARD.md` + a `TRAINING_HISTORY.md` Part 9 in the ledger
style: the 2×2 with CIs, the official number vs 0.827, wall times, geometry
(0.972 mpp) stated on every figure, the site caveat (C16 is ~two centers;
FM embeddings encode scanner fingerprints — CAMELYON17 is the multi-center
stress test we have NOT run), and any guard refusals with resolutions.
Commit `results/c16_abmil/` (protocols, aggregate official JSON, attention
PNGs) and `data/manifests/c16_abmil_v1/official_queries.jsonl` — the ledger is
the audit trail proving the shot was spent once.

Round 2 decision (record, don't run): Virchow2 at 0.972 and/or the chosen
encoder at 0.5 mpp — each needs a new pre-registration AND a deliberate budget
change in `dataset.json`. Note for round 2: the tile-extraction geometry fix
(2026-09-07) only changes behavior when the pyramid level's mpp differs from
the target by >2%, which at 0.972 never happens — so a 0.5 mpp round is the
first run that actually exercises the resampling path.

## Reporting back

Update `TRAINING_HISTORY.md`, commit result JSONs + CARD + attention PNGs +
the query ledger, push. Sienna's session will pull and review, and she'll take
the result to Dean.
