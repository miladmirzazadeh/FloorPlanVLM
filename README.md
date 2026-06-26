# FloorPlanVLM — reproducible training (RunPod / A100)

A clean, **resumable, crash-safe** reimplementation of the training pipeline from
**FloorPlanVLM: A Vision-Language Model for Floorplan Vectorization**
([arXiv:2602.06507](https://arxiv.org/abs/2602.06507)), built to run on a single
**A100 80GB** on RunPod with **two commands** and survive pod crashes / credit
exhaustion / a closed laptop.

It fine-tunes **Qwen2.5-VL-3B** to turn a raster floor plan into structured JSON
(walls, doors, windows, rooms) using the paper's three-stage recipe:

| Stage | What | Script |
|------|------|--------|
| 1 + 2 | Supervised fine-tuning (LoRA) on CubiCasa5K | `src/train_sft.py` |
| 3 | GRPO geometric alignment (reward = `0.1·R_val + 0.5·R_ext + α·0.4·R_int`) | `src/train_grpo.py` |

> **Scope honesty.** The paper's 92.52% wall-IoU was trained by Beike on *proprietary*
> data (Floorplan-2M / HQ-300K) on 32×H200 with full fine-tuning. That data is not
> public. This repo reproduces the **method** on public **CubiCasa5K** with **LoRA** on
> one GPU — great for learning/iterating, not for matching the paper's headline number.

---

## TL;DR — run on RunPod

1. Start an **A100 80GB** pod (RunPod PyTorch template). Add a persistent **Network
   Volume** mounted at `/workspace`. Set two **environment variables** on the pod:
   `HF_TOKEN` (a write token) and `HF_USER` (your HF username). *(Optional: `SMOKE_TEST=1`
   for a 5-minute dry run first.)*
2. In the pod's web terminal, paste **one** line:

   ```bash
   cd /workspace && git clone https://github.com/miladmirzazadeh/FloorPlanVLM.git 2>/dev/null; cd FloorPlanVLM && git pull -q; bash scripts/runpod_bootstrap.sh
   ```

That's it. Training now runs **in the background**. Watch it with:

```bash
bash scripts/status.sh
```

You can close the terminal and shut your laptop — it keeps going.

**If the pod dies or you run out of credits:** start a new A100 pod (reattach the same
volume if you have one, set the same env vars) and paste the **exact same one-liner**.
It detects existing checkpoints on the Hub and **resumes from where it left off**.

---

## How "never lose work" is implemented

- **Durable autosave → Hugging Face Hub.** During training, full checkpoints
  (adapter + optimizer + scheduler + RNG state) stream to two private HF repos
  (`<HF_USER>/floorplan-vlm-sft` and `-grpo`) via `hub_strategy="all_checkpoints"`.
  The Hub is the source of truth, so work survives even a *terminated* pod (which wipes
  local disk). HF Hub is used instead of Kaggle here because the whole `transformers`
  stack pushes/pulls checkpoints to it natively — see [docs/RUNPOD.md](docs/RUNPOD.md)
  if you specifically want a Kaggle mirror.
- **Resume.** On (re)start each stage pulls the latest Hub checkpoint and calls
  `trainer.train(resume_from_checkpoint=...)`.
- **Best model.** SFT keeps a held-out eval split and `load_best_model_at_end`, so the
  final pushed adapter is the best-eval one, not just the last step.
- **Idempotent stages.** When a stage finishes it writes a `FINISHED` marker to its Hub
  repo; re-running it (e.g. on a fresh pod) becomes a no-op, so GRPO won't redo SFT.
- **Detached + auto-restart.** `runpod_bootstrap.sh` launches a watchdog
  (`run_pipeline.sh`) with `setsid nohup`, so it's independent of your SSH session; the
  watchdog restarts any stage that crashes and resumes it.

## Configuration

Everything is environment-driven — **no need to edit code on the pod**. See
[`.env.example`](.env.example). Common knobs: `SMOKE_TEST`, `MAX_SAMPLES`,
`NUM_EPOCHS_SFT`, `GRPO_MAX_SAMPLES`, `NUM_GENERATIONS`, `RUN_GRPO`, and
`DATA_DIR`/`OUTPUT_DIR_*` (point these at `/workspace/...` to use the persistent volume).

## Inference

```bash
python -m src.infer path/to/floorplan.png                      # uses your GRPO adapter
python -m src.infer path/to/floorplan.png <HF_USER>/floorplan-vlm-sft   # SFT only
```

## Evaluation

Score an adapter on the held-out split with the paper's metrics (validity, external-wall
IoU, room IoU/F1, room-label F1, opening F1, wall-count MAE):

```bash
python -m src.eval --adapter <HF_USER>/floorplan-vlm-grpo --limit 100
DATASETS=synth python -m src.eval        # measure one dataset's contribution
```

Same held-out split as training (seed 42), so deltas across datasets/stages are
comparable — see [docs/RUNPOD.md §13](docs/RUNPOD.md). Metric math: [`src/metrics.py`](src/metrics.py).

## Layout

```
src/   config.py · prompts.py · taxonomy.py · geometry.py · data.py · data_msd.py · data_struct3d.py · data_synth.py · rewards.py · metrics.py · hub_utils.py · train_sft.py · train_grpo.py · infer.py · eval.py
scripts/  runpod_bootstrap.sh · run_pipeline.sh · status.sh · stop.sh
docs/   RUNPOD.md   (pod setup, cost, multi-dataset, curved walls, scaling to Qwen-30B, troubleshooting)
```

Curved walls are supported end-to-end via a signed-sagitta `curvature` field — the GRPO
IoU reward and all renderers are arc-aware ([`src/geometry.py`](src/geometry.py)); set
`FIT_CURVES=1` to have parsers fit arcs to curved geometry ([docs/RUNPOD.md §11](docs/RUNPOD.md)).

## More data (optional): multi-dataset training

CubiCasa-only is the default. Mix in more data by setting `DATASETS` — sources are
harmonized (coords → 1024, unified room labels in [`src/taxonomy.py`](src/taxonomy.py),
openings → `center+width`) and shuffled together:

| Dataset | `DATASETS` token | Get it | Why |
|---|---|---|---|
| CubiCasa5K | `cubicasa` | auto (Zenodo) | real, default baseline |
| **Structured3D** | `struct3d` | **auto** (39 MB annotations only) | **synthetic → pixel-perfect**, clean room types, slanted walls (paper's HQ trick) |
| MSD | `msd` | manual ([4TU](https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9)) | real, complex multi-unit geometry |
| **synth-floorseg** | `synth` | manual (Kaggle zip) | **richest**: synthetic CAD renders + explicit wall centerlines/thickness, **real curved-wall arcs**, room types |

E.g. `DATASETS=cubicasa,struct3d,synth`. Both the Structured3D and synth parsers are
validated on real data (S3D 30/30 scenes; synth incl. curved walls — a κ=1.0 semicircle
reconstructed exactly). Full steps, per-dataset caps for the mix ratio, and one-file
verification commands are in [docs/RUNPOD.md §9–12](docs/RUNPOD.md).

## Provenance & license

- Method: FloorPlanVLM (arXiv:2602.06507), Liu et al., Beike.
- Data/geometry parsing & reward shaping adapted from the community reference
  [`manitocross/floorplan-vlm-training`](https://huggingface.co/manitocross/floorplan-vlm-training)
  and the [`mudasir13cs`](https://github.com/mudasir13cs) SFT/GRPO repos, then
  restructured here for env-driven config, Hub-checkpoint resume, real-data GRPO, and
  detached crash-safe orchestration.
- **CubiCasa5K is CC BY-NC 4.0 — non-commercial research use only.**
