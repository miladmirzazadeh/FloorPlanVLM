# RunPod guide — step by step

## 0. What you need
- A RunPod account with credits.
- A Hugging Face account + a **write** token: https://huggingface.co/settings/tokens
- Your HF username (e.g. `miladmirzazadeh`).

## 1. Create the pod
1. **Deploy → Pods**, pick **A100 80GB** (PCIe or SXM). Community Cloud is cheapest
   (~$1.2/hr); Secure Cloud is steadier (~$1.9/hr).
2. Template: **RunPod PyTorch 2.x** (CUDA 12.x).
3. **Storage:** add a **Network Volume** (~80 GB) mounted at `/workspace`. This is what
   lets a *stopped* pod keep its data; for a *terminated* pod the Hub checkpoints are
   your safety net.
4. **Environment variables** (Edit Pod → Environment): set
   - `HF_TOKEN = hf_xxx...`
   - `HF_USER = miladmirzazadeh`
   - *(optional first run)* `SMOKE_TEST = 1`
   - *(recommended)* `DATA_DIR=/workspace/cubicasa_data`,
     `OUTPUT_DIR_SFT=/workspace/outputs/sft`, `OUTPUT_DIR_GRPO=/workspace/outputs/grpo`
   Setting these as pod env vars means every resume is **zero typing** beyond the one-liner.

## 2. Launch (one line)
Open the **Web Terminal** (or SSH) and paste:

```bash
cd /workspace && git clone https://github.com/miladmirzazadeh/FloorPlanVLM.git 2>/dev/null; cd FloorPlanVLM && git pull -q; bash scripts/runpod_bootstrap.sh
```

The bootstrap installs deps once, logs into HF, and launches training detached. Then:

```bash
bash scripts/status.sh     # follow the live log; Ctrl-C only stops watching
```

Close the terminal / laptop whenever — training continues on the pod.

## 3. Smoke test first (strongly recommended)
With `SMOKE_TEST=1` the whole pipeline runs on ~40 plans in a few minutes and pushes a
tiny checkpoint to the Hub. Use it to confirm: HF auth works, data downloads, SFT and
GRPO both start, and a checkpoint appears at
`https://huggingface.co/<HF_USER>/floorplan-vlm-sft`. Then remove `SMOKE_TEST` (or set
it to `0`), wipe the smoke markers, and relaunch for the real run:

```bash
rm -f state/*.done                      # forget the tiny smoke run
# (optionally delete the smoke checkpoints in the HF repos from the website)
bash scripts/runpod_bootstrap.sh
```

## 4. Resuming after a crash / out-of-credits
Spin up a fresh A100 pod. If you kept the Network Volume, reattach it. Set the same env
vars. Paste the **same one-liner** from step 2. Behavior:
- Volume survived → resumes from local checkpoint instantly.
- Volume gone → each stage pulls its latest checkpoint from the Hub and continues; a
  stage already marked `FINISHED` is skipped.

## 5. Cost & time (CubiCasa5K, LoRA, 1×A100 80GB)
| Stage | ~Wall-clock | ~Cost @ $1.5/hr |
|------|------------|------------------|
| SFT (2 epochs, ~5k plans) | 4–12 h | $6–18 |
| GRPO (500 plans, G=4) | 4–12 h | $6–18 |
| **Total** | **~10–24 h** | **~$15–35** |

Cut cost: lower `GRPO_MAX_SAMPLES`, set `NUM_EPOCHS_SFT=1`, or `RUN_GRPO=false`.

## 6. Auto-start on boot (optional, fully hands-off)
Instead of pasting the one-liner each time, set the pod's **Container Start Command** to:

```bash
bash -lc 'cd /workspace && git clone https://github.com/miladmirzazadeh/FloorPlanVLM.git 2>/dev/null; cd FloorPlanVLM && git pull -q && bash scripts/runpod_bootstrap.sh'
```

Now any new pod with your env vars resumes training automatically on boot.

## 7. Scaling to Qwen-30B (Qwen3-VL-30B-A3B, MoE)
The MoE is memory-heavy but compute-light (≈3B active). To try it:
```bash
MODEL_ID=Qwen/Qwen3-VL-30B-A3B-Instruct
```
plus, for fitting on GPU, load in 4-bit (QLoRA). SFT QLoRA fits one A100 80GB; **GRPO
needs ~2× A100 80GB** because it holds the policy + a generation copy. Expect ~4–8× the
cost of the 3B run. See the cost/quality discussion in the project chat — for CubiCasa5K
(only ~5k plans) the 3B is usually the right call; 30B mostly helps JSON validity and
non-Manhattan geometry, and overfits faster on small data.

> Note: the `Qwen2_5_VLForConditionalGeneration` class in the scripts is specific to
> Qwen2.5-VL. For Qwen3-VL, switch to `AutoModelForVision2Seq` / the Qwen3-VL class and
> bump `transformers`. This repo's defaults target the paper's 3B base.

## 8. Troubleshooting
- **CUDA OOM (SFT):** lower `MAX_LEN_SFT` (e.g. 3072) or raise `GRAD_ACCUM_SFT`.
- **CUDA OOM (GRPO):** lower `NUM_GENERATIONS` (e.g. 2) or `MAX_COMPLETION_LENGTH`.
- **TRL/transformers API mismatch** (e.g. `SFTConfig`/`GRPOConfig` rejects an arg):
  pin a known-good set, then rerun bootstrap:
  ```bash
  pip install -q "transformers==4.51.3" "trl==0.16.1" "peft==0.14.0" "accelerate==1.4.0"
  rm -f state/.deps_installed   # so bootstrap won't reinstall over your pins... 
  ```
  (the marker only gates the *bulk* install; pinned versions you set manually stay.)
- **Hub push is slow:** raise `SAVE_STEPS_SFT` / `SAVE_STEPS_GRPO` to checkpoint less often.
- **GRPO can't find the SFT adapter:** make sure SFT finished (a `FINISHED` file exists in
  `<HF_USER>/floorplan-vlm-sft`); GRPO loads + merges that adapter before training.

## 9. Adding MSD (Modified Swiss Dwellings) — multi-dataset training
MSD adds 5.3K real, complex, multi-unit European layouts (thick exterior walls,
irregular rooms, shared corridors). It's **opt-in**, so it never affects a
CubiCasa-only run.

**Get the data** (one-time): download the MSD training archive from
[4TU.ResearchData](https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9)
(CC BY 4.0, ~4.7 GB) and extract it onto the volume so that `full_out/*.npy` exists:
```bash
# after downloading msd_train.zip to /workspace:
mkdir -p /workspace/msd_data && unzip -q /workspace/msd_train.zip -d /workspace/msd_data
```

**Verify the conversion on ONE file first** (the parser infers class indices from
MSD's `ROOM_NAMES` order — eyeball one sample before a full run):
```bash
MSD_DIR=/workspace/msd_data python -m src.data_msd /workspace/msd_data
# prints walls/rooms/openings counts + saves *_debug.png (red = reconstructed walls).
# If it says "no rooms detected", the class indices in src/taxonomy.py need adjusting
# to match the unique values printed.
```

**Enable it** by setting two env vars, then launch as usual:
```bash
DATASETS=cubicasa,msd
MSD_DIR=/workspace/msd_data
# optional: MSD_MAX_SAMPLES=2000  to cap / tune the mix ratio vs CubiCasa
```

**How harmonization is handled** (the things that matter when mixing datasets):
- *Coordinates*: every dataset is normalized to longest-edge = 1024, image resized to
  match — so coords are pixel-aligned and on one grid across datasets.
- *Taxonomy*: both datasets map onto the ~14 unified labels in `src/taxonomy.py`
  (e.g. CubiCasa "Hall" and MSD "Corridor" → `corridor`).
- *Openings*: reduced to the canonical `center + width` nested under the parent wall.

**Representation caveat (important):** MSD's graph omits walls, so we rebuild the
wall-centric schema from the `full_out` segmentation mask (rooms → polygon edges →
deduped walls; wall *thickness* is estimated from the Structure mask, not exact).
MSD mainly teaches **geometric complexity**; room *type* is intentionally not colour-
leaked into the rendered input, so type labels from MSD are weaker supervision than
its geometry. Mix it with CubiCasa rather than training on MSD alone.

## 10. Adding Structured3D — synthetic, pixel-perfect (the paper's HQ trick)
Structured3D is **synthetic**, so its geometry is exact: rendering a 2D floor plan from
it gives **pixel-perfect (image, JSON) pairs** — the single most impactful public proxy
for the paper's HQ-300K subset. Unlike MSD it has clean **vector** rooms AND real
**room-type** labels, plus non-Manhattan (slanted) walls.

**You only need the ~39 MB structure-annotation zip — NOT the 50+ multi-GB image zips**
(those are 3D renders we don't use; we draw our own top-down plans). It's
**auto-downloaded** on first use, so just enable it:
```bash
DATASETS=cubicasa,struct3d         # or: cubicasa,msd,struct3d
# S3D_DIR=/workspace/s3d_data       # where the annotation zip extracts (default ./s3d_data)
# S3D_MAX_SAMPLES=2000              # cap / tune the mix (3500 scenes total)
```

**Verify one scene** (validated on real data here — 30/30 scenes converted):
```bash
python -m src.data_struct3d /workspace/s3d_data    # prints counts + saves *_debug.png
```

**How it's converted** (`src/data_struct3d.py`): each scene's `annotation_3d.json` gives
3D junctions/planes; we project floor planes to 2D → room polygons (+ type), derive walls
from room edges (deduped), and project door/window planes → `center+width` openings.
Coordinates normalized to longest-edge 1024; image rendered to match → pixel-aligned.
Wall *thickness* is nominal (S3D walls are idealized zero-thickness planes).

**License:** Structured3D is for **non-commercial research**; make sure you've accepted
its [terms](https://structured3d-dataset.org/) before downloading. (Note: since CubiCasa
is also non-commercial, the whole project is already non-commercial.)

### Recommended data ladder
1. **CubiCasa only** — validate the pipeline (Stage 1 → Stage 2).
2. **+ struct3d** — biggest quality lever: pixel-perfect data + clean room types.
3. **+ msd** — adds real, complex multi-unit geometry for robustness.
Use `S3D_MAX_SAMPLES` / `MSD_MAX_SAMPLES` / `MAX_SAMPLES` to control the mix ratio.

## 11. Curved walls
The `curvature` field (κ) is fully supported end-to-end: it's a signed **sagitta ratio**
`κ = 2·h/L` (0 = straight, ±1 = semicircle; sign = bulge direction; scale-invariant).

- **Reward & rendering are arc-aware always** (`src/geometry.py`): the GRPO IoU reward
  reconstructs the arc, and every renderer draws it — so a correct curved prediction is
  rewarded, not penalized as if it were straight. This is on regardless of settings.
- **Producing curved labels is opt-in**: set `FIT_CURVES=1` to make the MSD/S3D parsers
  fit a single arc to smooth multi-vertex polygon runs (instead of many short straight
  edges). Default is `0` (off) so the validated straight-wall parsers are byte-identical.

Reality check: CubiCasa/MSD/S3D are mostly Manhattan/slanted (straight). The **synth**
dataset below is the exception — it ships explicit curved-wall arcs, so curvature is
learned from real labels there (no `FIT_CURVES` needed for it).

## 12. Adding synth-floorseg (your own synthetic set — the richest source)
This is your Vitruev generator's 10k synthetic plans: **rendered CAD images** + explicit
vector geometry (wall centerlines + `thickness_mm` + **`arc`** for curved walls + room
types + openings). It's the closest public-ish proxy to the paper's HQ data and the only
source with genuine curved-wall supervision.

**Get the data onto the pod**: it's your Kaggle dataset `synth-floorseg` (a zip of
`configs/`, `rich_json/`, `images/`). Transfer it (Kaggle, `runpodctl send`, or cloud
drive) and unzip so those three folders exist under one dir, then point `SYNTH_DIR` at it:
```bash
unzip -q synth_floorseg.zip -d /workspace/synth_data
DATASETS=cubicasa,synth          # or struct3d,synth, etc.
SYNTH_DIR=/workspace/synth_data
# optional: SYNTH_MAX_SAMPLES=4000  to cap / tune the mix (10k available)
```

**Verify one plan** (validated locally — incl. a κ=1.0 semicircle reconstructed exactly):
```bash
python -m src.data_synth /workspace/synth_data   # prints walls/curved/rooms + *_debug.png
```

**How it's converted** (`src/data_synth.py`): solves the mm→px affine from `rich_json`
same-point pairs (the generator's exact transform), maps wall centerlines + room polygons
+ openings to pixels, normalizes to longest-edge 1024, resizes the rendered PNG to match
(pixel-aligned). Curved walls come straight from the `arc` field → exact `curvature`.
Room types map to the unified taxonomy; openings → `center+width` on the nearest wall.

## 13. Evaluating (measuring what each dataset/stage adds)
After training, score the adapter on the held-out split with the paper's metrics:
```bash
python -m src.eval                                  # GRPO adapter, 100 held-out samples
python -m src.eval --adapter <user>/floorplan-vlm-sft --limit 200   # SFT only
python -m src.eval --adapter base                   # untrained baseline, for reference
DATASETS=synth python -m src.eval                   # eval on ONE dataset's holdout
```
It reports **validity rate, external-wall IoU, room IoU/F1, room-label F1, opening F1,
and wall-count MAE**, and writes `eval_results.json`. The split is the same one training
held out (`get_sft_datasets`, seed 42), so it's genuinely unseen.

How to read it: train with `DATASETS=cubicasa`, eval; then `DATASETS=cubicasa,synth`,
retrain, eval — the IoU/F1 deltas tell you what each dataset actually buys. Run after SFT
and again after GRPO to see the RL gain (the paper's big jump is in validity + ext-IoU).
Metric definitions live in `src/metrics.py`.
