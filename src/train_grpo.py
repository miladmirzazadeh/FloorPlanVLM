"""Stage 3: GRPO geometric alignment, starting from the SFT adapter.

Handoff fix vs. the reference: we load the base model, attach the SFT LoRA adapter,
`merge_and_unload()` so the SFT knowledge is baked into the weights, then train a FRESH
LoRA with GRPO. (The reference passed the adapter repo id straight in as the base model
+ a new peft_config, which double-wraps and may not load the adapter at all.)

Same resume/finish machinery as SFT: Hub checkpoints + resume + FINISHED marker.
"""
import os

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, PeftModel

from . import config, hub_utils
from .data import build_grpo_dataset
from .rewards import floorplan_reward


class ConsoleLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in logs.items() if k != "epoch"]
        print(f"  step {state.global_step:>6} | " + " | ".join(parts), flush=True)


def _load_sft_merged(use_gpu):
    mk = {"torch_dtype": torch.bfloat16 if use_gpu else torch.float32}
    if use_gpu:
        try:
            import flash_attn  # noqa: F401
            mk["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.MODEL_ID, **mk)
    print(f"[grpo] loading + merging SFT adapter from {config.REPO_SFT} ...")
    model = PeftModel.from_pretrained(base, config.REPO_SFT, token=config.HF_TOKEN or None)
    return model.merge_and_unload()


def main():
    if not config.RUN_GRPO:
        print("[grpo] RUN_GRPO is false — skipping Stage 3.")
        return

    hub_utils.hf_login()
    config.banner("GRPO (Stage 3)")

    if hub_utils.is_finished(config.REPO_GRPO):
        print(f"[grpo] {config.REPO_GRPO} already FINISHED — skipping.")
        return
    if not hub_utils.is_finished(config.REPO_SFT):
        print(f"[grpo] NOTE: {config.REPO_SFT} not marked FINISHED; will try to load it anyway.")
    hub_utils.ensure_repo(config.REPO_GRPO)

    use_gpu = torch.cuda.is_available()
    processor = AutoProcessor.from_pretrained(
        config.MODEL_ID, min_pixels=config.IMG_MIN_PIXELS, max_pixels=config.IMG_MAX_PIXELS
    )
    dataset = build_grpo_dataset(config.ANN_PATH, config.GRPO_MAX_SAMPLES)
    model = _load_sft_merged(use_gpu)

    peft_config = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGETS, lora_dropout=config.LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM",
    )

    args = GRPOConfig(
        output_dir=config.OUTPUT_DIR_GRPO,
        num_train_epochs=config.NUM_EPOCHS_GRPO,
        per_device_train_batch_size=config.BATCH_SIZE_GRPO,
        gradient_accumulation_steps=config.GRAD_ACCUM_GRPO,
        learning_rate=config.LR_GRPO,
        bf16=use_gpu, fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, logging_first_step=True, disable_tqdm=True,
        save_strategy="steps", save_steps=config.SAVE_STEPS_GRPO, save_total_limit=3,
        num_generations=config.NUM_GENERATIONS,
        max_prompt_length=config.MAX_PROMPT_LENGTH,
        max_completion_length=config.MAX_COMPLETION_LENGTH,
        temperature=config.GRPO_TEMPERATURE,
        scale_rewards=True,
        beta=config.KL_COEF,
        push_to_hub=True,
        hub_model_id=config.REPO_GRPO,
        hub_strategy="all_checkpoints",
        hub_private_repo=config.PRIVATE_REPOS,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[floorplan_reward],
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processor,
    )

    # ── resume ──
    hub_utils.pull_latest_checkpoint(config.REPO_GRPO, config.OUTPUT_DIR_GRPO)
    last = get_last_checkpoint(config.OUTPUT_DIR_GRPO) if os.path.isdir(config.OUTPUT_DIR_GRPO) else None
    print(f"[grpo] resume_from_checkpoint = {last}")
    trainer.train(resume_from_checkpoint=last)

    trainer.save_model(config.OUTPUT_DIR_GRPO)
    try:
        trainer.push_to_hub(commit_message="final GRPO adapter")
    except Exception as e:
        print(f"[grpo] trainer.push_to_hub failed ({e}); manual upload")
        hub_utils.upload_folder(config.REPO_GRPO, config.OUTPUT_DIR_GRPO)
    hub_utils.mark_finished(config.REPO_GRPO)
    print(f"[grpo] DONE -> https://huggingface.co/{config.REPO_GRPO}")


if __name__ == "__main__":
    main()
