# Synthetic Floor-Plan Generator (validity-gated)

Procedurally generates architectural floor plans as
**rendered PNG + `rich_json` (mm↔px) + YOLO labels + config JSON**, where every
emitted plan is guaranteed valid by a strict topological/semantic validator:

- watertight exterior, no floating walls, closed room loops, gap/overlap-free partition;
- every room reachable from an entrance through a door (connected plan);
- **no wall crosses an opening** (no door/window blocked by a wall);
- sensible, standard opening widths; curved/slanted/non-Manhattan footprints preserved.

📖 **Full spec:** see [`DATASET_CARD.md`](DATASET_CARD.md) — implementation,
artifact schemas, data distributions, validity guarantees, and the rendering
specification.

## Layout

```
config_generator.py     # stage 1: validity-gated config JSON (procedural engine)
validate_plan.py        # strict validator  (T1–T8, S1–S10, C1–C5)  — also a CLI
render_dataset.py       # stage 2: configs -> PNG + rich_json + YOLO labels
generate_dataset.py     # one-command orchestrator (stage1 + stage2, resumable)
generate_valid.py       # all-in-one small-batch generator + contact sheet
generator/              # rendering engine (DXF -> PNG, exporters)
kaggle/                 # ready-to-run Kaggle notebook
DATASET_CARD.md         # full implementation + data + rendering spec
requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
python generate_dataset.py --count 30000 --output dataset_valid --workers 8
```

Output:

```
dataset_valid/
  images/{train,val}/plan_XXXXX.png
  labels/{train,val}/plan_XXXXX.txt      # YOLO
  rich_json/plan_XXXXX_rich.json         # mm↔px opening records (pixel-aligned)
  configs/plan_XXXXX.json                # full geometry config
  data.yaml
```

Resumable: a PNG on disk implies its labels exist, so re-running continues where
it stopped. Validate any time:

```bash
python validate_plan.py dataset_valid/configs --json check.json   # expect 100% valid
python validate_plan.py dataset_valid/configs/plan_00001.json     # one plan
```

## Run on Kaggle (background) → save to a Kaggle Dataset

1. New Notebook → **File → Import Notebook** → upload
   `kaggle/floorplan_synthgen_kaggle.ipynb`. Settings: **Internet: On**,
   **Accelerator: None (CPU)**.
2. Edit the parameters cell: set `REPO_URL` to this repo and `DATASET_SLUG` to
   `your-username/floorplan-synth-30k`.
3. **Save Version → “Save & Run All (Commit)”** → runs unattended in the
   background and saves `/kaggle/working`.
4. Save to a dataset: **Output → New Dataset** (no token), or the API cell
   (needs `KAGGLE_USERNAME`/`KAGGLE_KEY` via *Add-ons → Secrets*).

### How long? (rendering is the bottleneck — **CPU only, a GPU does not help**)

Rendering is `ezdxf → matplotlib` rasterisation: **~3.5–4 s/plan/core**, dominated
by drawing line/hatch entities + `savefig` (DPI barely matters). So:

> **30k plans ≈ ~30 core-hours.** To finish in **~3 h you need ~10 cores in parallel.**

A Kaggle **CPU** notebook ≈ 4 cores. The renderer pins BLAS to 1 thread/worker so
those 4 cores actually scale (~0.9–1 plan/s/session). Options:

| approach | wall-clock (30k) |
|---|---|
| 1 CPU session (4 cores) | ~8–9 h → needs ~1 resume commit |
| **3 CPU sessions in parallel (shards)** | **~3 h** ✅ |
| 16–32-vCPU cloud VM | < 1 h (most reliable for a deadline) |

**To hit ~3 h on Kaggle:** set `N_SHARDS = 3` in the notebook and run **3 copies**
(`SHARD = 0,1,2`) *at the same time* — each renders 10k (~3 h) and pushes
`…-shard0/1/2`; attach all three when training. This needs Kaggle to let you run
3 concurrent sessions (check your quota) and a GPU is **not** wanted (use CPU).

- **Single job + resume:** `N_SHARDS = 1`; commit, and if it times out attach the
  previous output as an input and commit again (the resume cell copies it back in).
- Shards never collide: shard *k* uses indices `[k·count, (k+1)·count)`, so plan
  ids (`plan_10001…`) and batch files are globally unique and mergeable.
- **Configs are cheap** (minutes for 30k) — only rasterisation is slow. If your
  trainer can render on-the-fly, generate configs (`--skip-render`) and rasterise
  in the dataloader instead.

## Notes

- `--workers 0` = auto (min(cpu, 8)). Kaggle CPU = ~4 cores → `--workers 4`.
- `--max-dpi 150` caps the canvas (the engine samples 80–220 DPI); mainly bounds
  **file size** (~3–4 GB for 30k) — `rich_json` stays pixel-aligned to the PNG.
- `--start N` shards the index range; `--skip-config` / `--skip-render` run a
  single stage.
- Schema is identical to the original pipeline; `validate_plan.py` documents the
  exact T/S/C constraints.
