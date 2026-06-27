"""Supervised fine-tuning — paper-faithful 2-stage curriculum (§4.4).

Selected by SFT_STAGE (run_pipeline.sh runs 1 then 2):
  Stage 1 "Structural Grounding"  — new LoRA on diverse REAL data (STAGE1_DATASETS)
                                     -> generalized layout, not pixel precision.
  Stage 2 "Quality Annealing"     — CONTINUE the Stage-1 adapter on PIXEL-PERFECT
                                     synthetic data (STAGE2_DATASETS) -> watertight precision.

Both stages: resumable (Hub checkpoints + resume), FINISHED markers, best-on-eval.
"""
import os

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel

from . import config, hub_utils
from .data import get_sft_datasets


class ConsoleLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in logs.items() if k != "epoch"]
        print(f"  step {state.global_step:>6} | " + " | ".join(parts), flush=True)


def _load_base(use_gpu):
    mk = {"torch_dtype": torch.bfloat16 if use_gpu else torch.float32}
    if use_gpu:
        try:
            import flash_attn  # noqa: F401
            mk["attn_implementation"] = "flash_attention_2"
            print("[sft] using flash_attention_2")
        except Exception:
            print("[sft] flash-attn not installed; default attention")
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(config.MODEL_ID, **mk)


def main():
    hub_utils.hf_login()
    stage = config.SFT_STAGE
    if stage == 1:
        datasets, repo, out_dir, epochs, prev = (
            config.STAGE1_DATASETS, config.REPO_SFT1, config.OUTPUT_DIR_SFT1,
            config.NUM_EPOCHS_SFT1, None)
        label = "SFT Stage 1 — Structural Grounding"
    else:
        datasets, repo, out_dir, epochs, prev = (
            config.STAGE2_DATASETS, config.REPO_SFT, config.OUTPUT_DIR_SFT,
            config.NUM_EPOCHS_SFT2, config.REPO_SFT1)
        label = "SFT Stage 2 — Quality Annealing"

    config.banner(label)
    print(f"[sft] stage={stage} datasets={datasets} repo={repo} epochs={epochs} prev={prev}")

    if hub_utils.is_finished(repo):
        print(f"[sft] {repo} already FINISHED — skipping.")
        return
    hub_utils.ensure_repo(repo)

    use_gpu = torch.cuda.is_available()
    train_ds, eval_ds = get_sft_datasets(datasets)
    print(f"[sft] train={len(train_ds)}  eval={len(eval_ds) if eval_ds else 0}")

    px = ({"min_pixels": config.IMG_MIN_PIXELS, "max_pixels": config.IMG_MAX_PIXELS} if use_gpu
          else {"min_pixels": 64 * 28 * 28, "max_pixels": 256 * 28 * 28})
    processor = AutoProcessor.from_pretrained(config.MODEL_ID, **px)

    base = _load_base(use_gpu)
    if prev is None:
        model = base
        peft_config = LoraConfig(
            r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGETS, lora_dropout=config.LORA_DROPOUT,
            bias="none", task_type="CAUSAL_LM")
    else:
        print(f"[sft] continuing Stage-1 adapter {prev}")
        model = PeftModel.from_pretrained(base, prev, is_trainable=True)
        peft_config = None

    has_eval = eval_ds is not None
    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=config.BATCH_SIZE_SFT,
        gradient_accumulation_steps=config.GRAD_ACCUM_SFT,
        learning_rate=config.LR_SFT,
        warmup_steps=20 if use_gpu else 1,
        lr_scheduler_type="cosine",
        bf16=use_gpu, fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, logging_first_step=True, disable_tqdm=True,
        save_strategy="steps", save_steps=config.SAVE_STEPS_SFT, save_total_limit=3,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=config.SAVE_STEPS_SFT if has_eval else None,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss", greater_is_better=False,
        max_length=config.MAX_LEN_SFT,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        push_to_hub=True, hub_model_id=repo, hub_strategy="all_checkpoints",
        hub_private_repo=config.PRIVATE_REPOS, report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        peft_config=peft_config, processing_class=processor,
        callbacks=[ConsoleLogger()],
    )

    hub_utils.pull_latest_checkpoint(repo, out_dir)
    last = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    print(f"[sft] resume_from_checkpoint = {last}")
    trainer.train(resume_from_checkpoint=last)

    trainer.save_model(out_dir)
    try:
        trainer.push_to_hub(commit_message=f"final SFT stage {stage} adapter")
    except Exception as e:
        print(f"[sft] trainer.push_to_hub failed ({e}); manual upload")
        hub_utils.upload_folder(repo, out_dir)
    hub_utils.mark_finished(repo)
    print(f"[sft] stage {stage} DONE -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
