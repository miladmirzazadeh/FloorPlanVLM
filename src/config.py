"""Central, environment-driven configuration.

Everything is overridable via environment variables (or a .env file), so you never
have to edit Python on the RunPod box. Sensible defaults reproduce the FloorPlanVLM
recipe (arXiv:2602.06507) at the public-data / single-A100 scale.
"""
import os


def _s(key, default):
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def _i(key, default):
    v = os.environ.get(key)
    return int(v) if v not in (None, "") else default


def _f(key, default):
    v = os.environ.get(key)
    return float(v) if v not in (None, "") else default


def _b(key, default):
    return _s(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _opt_i(key, default):
    """Integer that may be 'none' to mean Python None (e.g. MAX_SAMPLES=none -> all)."""
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    if v.strip().lower() == "none":
        return None
    return int(v)


# ── Identity / Hub ────────────────────────────────────────────────────────────
HF_TOKEN = _s("HF_TOKEN", "")
HF_USER = _s("HF_USER", "")  # your HuggingFace username; used to derive repo ids
MODEL_ID = _s("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
PRIVATE_REPOS = _b("PRIVATE_REPOS", True)


def _derive_repo(stage):
    explicit = _s("HF_REPO_" + stage.upper(), "")
    if explicit:
        return explicit
    if HF_USER:
        return f"{HF_USER}/floorplan-vlm-{stage}"
    # No user set — leave a placeholder; hub_utils will warn loudly.
    return f"floorplan-vlm-{stage}"


REPO_SFT1 = _derive_repo("sft1")  # Stage-1 (structural grounding) adapter
REPO_SFT = _derive_repo("sft")    # Stage-2 (quality annealing) = final SFT adapter
REPO_GRPO = _derive_repo("grpo")  # Stage-3 (GRPO) adapter

# ── Datasets: paper-faithful curriculum (§4.4 Progressive Training) ─────────────
# Stage 1 "Structural Grounding" (paper's Floorplan-2M): large, DIVERSE, REAL data —
#   coordinate-noisy; learns generalized layout/topology, NOT pixel precision.
STAGE1_DATASETS = [d.strip() for d in _s("STAGE1_DATASETS", "cubicasa,msd").split(",") if d.strip()]
# Stage 2 "Quality Annealing" (paper's Floorplan-HQ-300K, ~93% synthetic-rendered):
#   PIXEL-PERFECT synthetic data (rendered from exact vectors) for watertight precision.
STAGE2_DATASETS = [d.strip() for d in _s("STAGE2_DATASETS", "synth,struct3d").split(",") if d.strip()]
# Which SFT stage this process runs (run_pipeline.sh sets it: 1 then 2).
SFT_STAGE = _i("SFT_STAGE", 1)
# Union — used by eval / any non-staged path.
_UNION = list(dict.fromkeys(STAGE1_DATASETS + STAGE2_DATASETS))
DATASETS = [d.strip() for d in _s("DATASETS", ",".join(_UNION)).split(",") if d.strip()]
# GRPO runs on the pixel-perfect Stage-2 data (paper applies GRPO after HQ annealing).
_grpo_env = _s("GRPO_DATASETS", "")
GRPO_DATASETS = ([d.strip() for d in _grpo_env.split(",") if d.strip()] if _grpo_env
                 else list(STAGE2_DATASETS))

# ── Paths (persistent volume on RunPod is /workspace) ─────────────────────────
DATA_DIR = _s("DATA_DIR", "./cubicasa_data")
ANN_PATH = os.path.join(DATA_DIR, "annotations.json")  # combined, all datasets
ZENODO_URL = _s("ZENODO_URL", "https://zenodo.org/record/2613548/files/cubicasa5k.zip?download=1")
OUTPUT_DIR_SFT1 = _s("OUTPUT_DIR_SFT1", "./outputs/sft1")
OUTPUT_DIR_SFT = _s("OUTPUT_DIR_SFT", "./outputs/sft")
OUTPUT_DIR_GRPO = _s("OUTPUT_DIR_GRPO", "./outputs/grpo")

# MSD (Modified Swiss Dwellings) — download the train archive from 4TU and extract
# so that <MSD_DIR>/.../full_out/*.npy exists (see docs/RUNPOD.md).
MSD_DIR = _s("MSD_DIR", "./msd_data")
MSD_RENDER_DIR = _s("MSD_RENDER_DIR", os.path.join(MSD_DIR, "rendered"))
MSD_MAX_SAMPLES = _opt_i("MSD_MAX_SAMPLES", None)

# Structured3D — we only need the ~39MB structure-annotation zip (auto-downloaded).
S3D_DIR = _s("S3D_DIR", "./s3d_data")
S3D_RENDER_DIR = _s("S3D_RENDER_DIR", os.path.join(S3D_DIR, "rendered"))
S3D_ANNOT_URL = _s(
    "S3D_ANNOT_URL",
    "https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/Structured3D_annotation_3d.zip",
)
S3D_MAX_SAMPLES = _opt_i("S3D_MAX_SAMPLES", None)
S3D_WALL_THICKNESS = _i("S3D_WALL_THICKNESS", 10)  # nominal (S3D walls are idealized planes)

# synth-floorseg — the user's own 10k synthetic set (Kaggle: synth-floorseg).
# Point SYNTH_DIR at the unzipped folder containing configs/ rich_json/ images/.
SYNTH_DIR = _s("SYNTH_DIR", "./synth_data")
SYNTH_RENDER_DIR = _s("SYNTH_RENDER_DIR", os.path.join(SYNTH_DIR, "rendered"))
SYNTH_MAX_SAMPLES = _opt_i("SYNTH_MAX_SAMPLES", None)
# The generator has topology artifacts (unclosed loops, floating walls). Keep only
# topologically-clean plans for the pixel-perfect tier — bad GT would poison Stage 2
# and the GRPO closure reward. Quality over quantity.
SYNTH_TOPO_FILTER = _b("SYNTH_TOPO_FILTER", True)
SYNTH_TOPO_MIN_JUNCTION = _f("SYNTH_TOPO_MIN_JUNCTION", 0.8)

# ── Data shaping ──────────────────────────────────────────────────────────────
MAX_JSON_CHARS = _i("MAX_JSON_CHARS", 10000)
MAX_SAMPLES = _opt_i("MAX_SAMPLES", None)  # None = all ~5k plans
EVAL_RATIO = _f("EVAL_RATIO", 0.03)        # held-out split for best-model tracking
# Detect curved walls (fit a single arc to smooth polygon runs) instead of many
# short straight edges. Experimental; off by default so validated parsers are
# byte-identical. Curvature is always honored in the schema/reward/render.
FIT_CURVES = _b("FIT_CURVES", False)

# Walls-only mode: the model extracts ONLY walls (a downstream VLM handles rooms/
# openings/semantics). Strips rooms+openings from all targets and uses a walls-only
# prompt -> shorter outputs, faster/cheaper, better wall accuracy, every dataset usable.
WALLS_ONLY = _b("WALLS_ONLY", False)

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R = _i("LORA_R", 16)
LORA_ALPHA = _i("LORA_ALPHA", 32)
LORA_DROPOUT = _f("LORA_DROPOUT", 0.05)
LORA_TARGETS = _s(
    "LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).split(",")

# ── SFT (paper Stages 1+2) ────────────────────────────────────────────────────
NUM_EPOCHS_SFT1 = _i("NUM_EPOCHS_SFT1", 2)   # paper: 2 epochs on Floorplan-2M (grounding)
NUM_EPOCHS_SFT2 = _i("NUM_EPOCHS_SFT2", 3)   # paper: 10 epochs on HQ-300K (scaled to our data size)
NUM_EPOCHS_SFT = _i("NUM_EPOCHS_SFT", 2)     # legacy single-stage fallback
BATCH_SIZE_SFT = _i("BATCH_SIZE_SFT", 1)
GRAD_ACCUM_SFT = _i("GRAD_ACCUM_SFT", 8)
LR_SFT = _f("LR_SFT", 2e-5)
SAVE_STEPS_SFT = _i("SAVE_STEPS_SFT", 200)   # checkpoint+push cadence
MAX_LEN_SFT = _i("MAX_LEN_SFT", 4096)

# ── GRPO (paper Stage 3) ──────────────────────────────────────────────────────
RUN_GRPO = _b("RUN_GRPO", True)
GRPO_MAX_SAMPLES = _opt_i("GRPO_MAX_SAMPLES", 500)  # RL is slow; subset is normal
NUM_EPOCHS_GRPO = _i("NUM_EPOCHS_GRPO", 1)
BATCH_SIZE_GRPO = _i("BATCH_SIZE_GRPO", 1)
GRAD_ACCUM_GRPO = _i("GRAD_ACCUM_GRPO", 4)
LR_GRPO = _f("LR_GRPO", 1e-6)
NUM_GENERATIONS = _i("NUM_GENERATIONS", 4)             # G completions per prompt
MAX_COMPLETION_LENGTH = _i("MAX_COMPLETION_LENGTH", 3072)
MAX_PROMPT_LENGTH = _i("MAX_PROMPT_LENGTH", 2048)      # must exceed image-token count
KL_COEF = _f("KL_COEF", 0.01)
GRPO_TEMPERATURE = _f("GRPO_TEMPERATURE", 0.7)
SAVE_STEPS_GRPO = _i("SAVE_STEPS_GRPO", 100)

# ── Smoke test: tiny, fast end-to-end validation of the whole loop+resume ──────
SMOKE_TEST = _b("SMOKE_TEST", False)
if SMOKE_TEST:
    MAX_SAMPLES = min(MAX_SAMPLES or 40, 40)
    MSD_MAX_SAMPLES = min(MSD_MAX_SAMPLES or 40, 40)
    S3D_MAX_SAMPLES = min(S3D_MAX_SAMPLES or 40, 40)
    SYNTH_MAX_SAMPLES = min(SYNTH_MAX_SAMPLES or 40, 40)
    NUM_EPOCHS_SFT1 = 1
    NUM_EPOCHS_SFT2 = 1
    NUM_EPOCHS_SFT = 1
    SAVE_STEPS_SFT = 5
    GRPO_MAX_SAMPLES = min(GRPO_MAX_SAMPLES or 16, 16)
    NUM_EPOCHS_GRPO = 1
    SAVE_STEPS_GRPO = 5
    MAX_LEN_SFT = min(MAX_LEN_SFT, 2048)


def banner(stage):
    print("=" * 72)
    print(f"  FloorPlanVLM — {stage}")
    print(f"  model      : {MODEL_ID}")
    print(f"  repo_sft   : {REPO_SFT}")
    print(f"  repo_grpo  : {REPO_GRPO}")
    print(f"  smoke_test : {SMOKE_TEST}   max_samples: {MAX_SAMPLES}")
    print("=" * 72, flush=True)
