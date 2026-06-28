"""Supervised fine-tune Qwen3-VL-8B (LoRA) on the built {image, target} JSONL.

- one frozen system prompt + image -> minified [0,1000] walls JSON (the target)
- loss is computed ONLY on the target tokens (system/user/image tokens are masked)
- training context capped at config.MAX_SEQ_LEN (image + prompt + target)
- checkpoints autosave to OUTPUT_DIR_SFT and (if HF_USER+HF_TOKEN set) push to the Hub
  each save; training auto-resumes from the latest local checkpoint.

    python -m src.train_sft        # reads built/train.jsonl, built/val.jsonl
"""
import os
import json

import torch
from PIL import Image
from transformers import AutoProcessor, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

from . import config, prompts
from .augment import augment

try:                                            # prefer the exact class when present
    from transformers import Qwen3VLForConditionalGeneration as VLM
except Exception:                               # fall back to the generic resolver
    from transformers import AutoModelForImageTextToText as VLM

# native BF16 (see config.TORCH_DTYPE) — never silently fall to FP16
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
DTYPE = _DTYPES.get(config.TORCH_DTYPE, torch.bfloat16)


def _rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _messages(target):
    return [
        {"role": "system", "content": [{"type": "text", "text": prompts.SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompts.USER_PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]


def make_collator(processor):
    tok = processor.tokenizer
    tok.padding_side = "right"

    def collate(examples):
        texts, images = [], []
        for ex in examples:
            img = Image.open(ex["image"]).convert("RGB")
            if config.AUGMENT:
                img = augment(img)               # pixel-level only; preserves coords
            images.append(img)
            texts.append(processor.apply_chat_template(
                _messages(ex["target"]), tokenize=False, add_generation_prompt=False))
        batch = processor(text=texts, images=images, return_tensors="pt",
                          padding=True, truncation=True, max_length=config.MAX_SEQ_LEN)
        labels = batch["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        # mask the prompt+image span per row -> loss only on the assistant target.
        for row, ex in enumerate(examples):
            ptext = processor.apply_chat_template(
                _messages(ex["target"])[:-1], tokenize=False, add_generation_prompt=True)
            plen = processor(text=[ptext], images=[images[row]], return_tensors="pt",
                             truncation=True, max_length=config.MAX_SEQ_LEN)["input_ids"].shape[1]
            labels[row, :plen] = -100
        batch["labels"] = labels
        return batch

    return collate


def main():
    config.banner("SFT  Qwen3-VL-8B")
    train = _rows(os.path.join(config.BUILT_DATA, "train.jsonl"))
    print(f"[sft] train={len(train)} samples  max_seq_len={config.MAX_SEQ_LEN}")

    processor = AutoProcessor.from_pretrained(
        config.MODEL_ID, min_pixels=config.IMG_MIN_PIXELS, max_pixels=config.IMG_MAX_PIXELS)
    model = VLM.from_pretrained(config.MODEL_ID, torch_dtype=DTYPE)
    print(f"[sft] dtype={config.TORCH_DTYPE}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    push = bool(config.HF_USER and config.HF_TOKEN)
    args = TrainingArguments(
        output_dir=config.OUTPUT_DIR_SFT,
        num_train_epochs=config.NUM_EPOCHS_SFT,
        max_steps=config.MAX_STEPS if config.MAX_STEPS > 0 else -1,
        per_device_train_batch_size=config.BATCH_SIZE_SFT,
        gradient_accumulation_steps=config.GRAD_ACCUM_SFT,
        learning_rate=config.LR_SFT,
        bf16=(DTYPE == torch.bfloat16),
        fp16=(DTYPE == torch.float16),
        logging_steps=10,
        save_steps=config.SAVE_STEPS_SFT,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,            # keep our raw dict columns for the collator
        dataloader_num_workers=2,
        push_to_hub=push,
        hub_model_id=config.REPO_SFT if push else None,
        hub_strategy="all_checkpoints" if push else "every_save",
        hub_private_repo=config.PRIVATE_REPOS,
        hub_token=config.HF_TOKEN or None,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train,
                      data_collator=make_collator(processor))

    resume = get_last_checkpoint(config.OUTPUT_DIR_SFT) if os.path.isdir(config.OUTPUT_DIR_SFT) else None
    if resume:
        print(f"[sft] resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(config.OUTPUT_DIR_SFT)
    processor.save_pretrained(config.OUTPUT_DIR_SFT)
    if push:
        trainer.push_to_hub()
    print(f"[sft] done -> {config.OUTPUT_DIR_SFT}" + (f"  (+ Hub {config.REPO_SFT})" if push else ""))


if __name__ == "__main__":
    main()
