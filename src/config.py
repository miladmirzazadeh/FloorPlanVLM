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


REPO_SFT = _derive_repo("sft")    # checkpoints + final SFT adapter live here
REPO_GRPO = _derive_repo("grpo")  # checkpoints + final GRPO adapter live here

# ── Paths (persistent volume on RunPod is /workspace) ─────────────────────────
DATA_DIR = _s("DATA_DIR", "./cubicasa_data")
ANN_PATH = os.path.join(DATA_DIR, "annotations.json")
ZENODO_URL = _s("ZENODO_URL", "https://zenodo.org/record/2613548/files/cubicasa5k.zip?download=1")
OUTPUT_DIR_SFT = _s("OUTPUT_DIR_SFT", "./outputs/sft")
OUTPUT_DIR_GRPO = _s("OUTPUT_DIR_GRPO", "./outputs/grpo")

# ── Data shaping ──────────────────────────────────────────────────────────────
MAX_JSON_CHARS = _i("MAX_JSON_CHARS", 10000)
MAX_SAMPLES = _opt_i("MAX_SAMPLES", None)  # None = all ~5k plans
EVAL_RATIO = _f("EVAL_RATIO", 0.03)        # held-out split for best-model tracking

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R = _i("LORA_R", 16)
LORA_ALPHA = _i("LORA_ALPHA", 32)
LORA_DROPOUT = _f("LORA_DROPOUT", 0.05)
LORA_TARGETS = _s(
    "LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).split(",")

# ── SFT (paper Stages 1+2) ────────────────────────────────────────────────────
NUM_EPOCHS_SFT = _i("NUM_EPOCHS_SFT", 2)
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
