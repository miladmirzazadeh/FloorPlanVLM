"""Central, environment-driven configuration — SFT-only, Qwen3-VL-8B.

New plan (2026-06): a single Supervised Fine-Tune of Qwen3-VL-8B on our own data
(cubicasa5k + synth + msd + archcad), emitting a token-optimized, deterministic
[0,1000]-grid JSON. No GRPO, no verifier loop in the training path.

Everything is overridable via environment variables so you never edit Python on the pod.
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
    """Integer that may be 'none' -> Python None (e.g. MAX_SAMPLES=none -> all)."""
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    if v.strip().lower() == "none":
        return None
    return int(v)


# ── Identity / Hub ────────────────────────────────────────────────────────────
HF_TOKEN = _s("HF_TOKEN", "")
HF_USER = _s("HF_USER", "")
# Qwen3-VL 8B instruct. (Verify the exact Hub id for your account; override with MODEL_ID.)
MODEL_ID = _s("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
PRIVATE_REPOS = _b("PRIVATE_REPOS", True)


def _derive_repo(stage):
    explicit = _s("HF_REPO_" + stage.upper(), "")
    if explicit:
        return explicit
    return f"{HF_USER}/floorplan-vlm-{stage}" if HF_USER else f"floorplan-vlm-{stage}"


REPO_SFT = _derive_repo("sft")   # the one and only output adapter

# ── Datasets ──────────────────────────────────────────────────────────────────
# Single combined SFT corpus. All four converters emit the SAME canonical raw walls
# (start/end px + thickness + openings), which the shared pipeline normalizes, orders,
# sorts, and encodes identically — so the model sees one consistent target format.
DATASETS = [d.strip() for d in _s("DATASETS", "cubicasa,synth,msd,archcad").split(",") if d.strip()]

# ── Paths (persistent volume on RunPod is /workspace) ─────────────────────────
DATA_DIR = _s("DATA_DIR", "./cubicasa_data")
ZENODO_URL = _s("ZENODO_URL", "https://zenodo.org/record/2613548/files/cubicasa5k.zip?download=1")
OUTPUT_DIR_SFT = _s("OUTPUT_DIR_SFT", "./outputs/sft")
BUILT_DATA = _s("BUILT_DATA", "./built")  # cached built dataset (image refs + targets)

MSD_DIR = _s("MSD_DIR", "./msd_data")
MSD_MAX_SAMPLES = _opt_i("MSD_MAX_SAMPLES", None)

SYNTH_DIR = _s("SYNTH_DIR", "./synth_data")
SYNTH_MAX_SAMPLES = _opt_i("SYNTH_MAX_SAMPLES", None)
SYNTH_TOPO_FILTER = _b("SYNTH_TOPO_FILTER", True)
SYNTH_TOPO_MIN_JUNCTION = _f("SYNTH_TOPO_MIN_JUNCTION", 0.8)

# ArchCAD — real CAD line-drawing floor plans. Point ARCHCAD_DIR at the unzipped set.
ARCHCAD_DIR = _s("ARCHCAD_DIR", "./archcad_data")
ARCHCAD_MAX_SAMPLES = _opt_i("ARCHCAD_MAX_SAMPLES", None)

CUBICASA_MAX_SAMPLES = _opt_i("CUBICASA_MAX_SAMPLES", None)

# ── Data format (the part that decides success) ───────────────────────────────
GRID = _i("GRID", 1000)                       # normalize ALL coords to [0, GRID]
PAD_TO_SQUARE = _b("PAD_TO_SQUARE", True)     # pad (never distort) to square before scaling
ABBREVIATE = _b("ABBREVIATE", True)           # short keys (cl/th/op) + minified JSON
NEST_OPENINGS = _b("NEST_OPENINGS", True)     # openings live inside their wall object
COUNT_ANCHOR = _b("COUNT_ANCHOR", True)       # prepend {"n":N,...} as lightweight CoT
SORT_WALLS = _b("SORT_WALLS", True)           # exterior clockwise, then interior TL->BR
ORDER_ENDPOINTS = _b("ORDER_ENDPOINTS", True) # cl always x1<=x2 (tie: y1<=y2)
NEG_SAMPLE_FRAC = _f("NEG_SAMPLE_FRAC", 0.04) # 3-5% empty/garbage -> {"n":0,"walls":[]}

# image token budget for the vision encoder (keep modest so image+text fit MAX_SEQ_LEN)
IMG_MIN_PIXELS = _i("IMG_MIN_PIXELS", 256 * 28 * 28)
IMG_MAX_PIXELS = _i("IMG_MAX_PIXELS", 1024 * 28 * 28)

MAX_SAMPLES = _opt_i("MAX_SAMPLES", None)
EVAL_RATIO = _f("EVAL_RATIO", 0.03)

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R = _i("LORA_R", 16)
LORA_ALPHA = _i("LORA_ALPHA", 32)
LORA_DROPOUT = _f("LORA_DROPOUT", 0.05)
LORA_TARGETS = _s("LORA_TARGETS",
                  "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")

# ── SFT ───────────────────────────────────────────────────────────────────────
NUM_EPOCHS_SFT = _i("NUM_EPOCHS_SFT", 3)
BATCH_SIZE_SFT = _i("BATCH_SIZE_SFT", 1)
GRAD_ACCUM_SFT = _i("GRAD_ACCUM_SFT", 8)
LR_SFT = _f("LR_SFT", 1e-4)
SAVE_STEPS_SFT = _i("SAVE_STEPS_SFT", 200)
# Cap the training context. Minified targets keep JSON short; capping image+prompt+
# target to 8192 (or 4096) slashes VRAM and speeds training without losing precision.
MAX_SEQ_LEN = _i("MAX_SEQ_LEN", 8192)

# ── Smoke test ────────────────────────────────────────────────────────────────
SMOKE_TEST = _b("SMOKE_TEST", False)
if SMOKE_TEST:
    MAX_SAMPLES = min(MAX_SAMPLES or 40, 40)
    MSD_MAX_SAMPLES = min(MSD_MAX_SAMPLES or 40, 40)
    SYNTH_MAX_SAMPLES = min(SYNTH_MAX_SAMPLES or 40, 40)
    ARCHCAD_MAX_SAMPLES = min(ARCHCAD_MAX_SAMPLES or 40, 40)
    CUBICASA_MAX_SAMPLES = min(CUBICASA_MAX_SAMPLES or 40, 40)
    NUM_EPOCHS_SFT = 1
    SAVE_STEPS_SFT = 5
    MAX_SEQ_LEN = min(MAX_SEQ_LEN, 4096)


def banner(stage):
    print("=" * 72)
    print(f"  FloorPlanVLM (SFT-only) — {stage}")
    print(f"  model     : {MODEL_ID}")
    print(f"  datasets  : {DATASETS}")
    print(f"  grid      : 0..{GRID}   max_seq_len: {MAX_SEQ_LEN}   abbreviate: {ABBREVIATE}")
    print(f"  repo_sft  : {REPO_SFT}")
    print(f"  smoke     : {SMOKE_TEST}   max_samples: {MAX_SAMPLES}")
    print("=" * 72, flush=True)
