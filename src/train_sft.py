"""Stage 1+2: Supervised fine-tuning of Qwen2.5-VL on CubiCasa5K (LoRA).

Resumable & crash-safe:
  * checkpoints stream to the Hub during training (hub_strategy='all_checkpoints');
  * on (re)start we pull the latest Hub checkpoint and resume_from_checkpoint;
  * a FINISHED marker on the Hub makes the whole stage a no-op once complete, so the
    watchdog can re-run this safely on a fresh pod.
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
from peft import LoraConfig

from . import config, hub_utils
from .data import get_sft_datasets


class ConsoleLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in logs.items() if k != "epoch"]
        print(f"  step {state.global_step:>6} | " + " | ".join(parts), flush=True)


def main():
    hub_utils.hf_login()
    config.banner("SFT (Stages 1+2)")

    if hub_utils.is_finished(config.REPO_SFT):
        print(f"[sft] {config.REPO_SFT} already FINISHED — skipping.")
        return
    hub_utils.ensure_repo(config.REPO_SFT)

    use_gpu = torch.cuda.is_available()
    train_ds, eval_ds = get_sft_datasets()
    print(f"[sft] train={len(train_ds)}  eval={len(eval_ds) if eval_ds else 0}")

    px = ({"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28} if use_gpu
          else {"min_pixels": 64 * 28 * 28, "max_pixels": 256 * 28 * 28})
    processor = AutoProcessor.from_pretrained(config.MODEL_ID, **px)

    mk = {"torch_dtype": torch.bfloat16 if use_gpu else torch.float32}
    if use_gpu:
        try:
            import flash_attn  # noqa: F401
            mk["attn_implementation"] = "flash_attention_2"
            print("[sft] using flash_attention_2")
        except Exception:
            print("[sft] flash-attn not installed; default attention")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.MODEL_ID, **mk)

    peft_config = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGETS, lora_dropout=config.LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM",
    )

    has_eval = eval_ds is not None
    args = SFTConfig(
        output_dir=config.OUTPUT_DIR_SFT,
        num_train_epochs=config.NUM_EPOCHS_SFT,
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
        push_to_hub=True,
        hub_model_id=config.REPO_SFT,
        hub_strategy="all_checkpoints",
        hub_private_repo=config.PRIVATE_REPOS,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        peft_config=peft_config, processing_class=processor,
        callbacks=[ConsoleLogger()],
    )

    # ── resume ──
    hub_utils.pull_latest_checkpoint(config.REPO_SFT, config.OUTPUT_DIR_SFT)
    last = get_last_checkpoint(config.OUTPUT_DIR_SFT) if os.path.isdir(config.OUTPUT_DIR_SFT) else None
    print(f"[sft] resume_from_checkpoint = {last}")
    trainer.train(resume_from_checkpoint=last)

    # ── finalize (model is the best checkpoint when has_eval) ──
    trainer.save_model(config.OUTPUT_DIR_SFT)
    try:
        trainer.push_to_hub(commit_message="final SFT adapter")
    except Exception as e:
        print(f"[sft] trainer.push_to_hub failed ({e}); manual upload")
        hub_utils.upload_folder(config.REPO_SFT, config.OUTPUT_DIR_SFT)
    hub_utils.mark_finished(config.REPO_SFT)
    print(f"[sft] DONE -> https://huggingface.co/{config.REPO_SFT}")


if __name__ == "__main__":
    main()
