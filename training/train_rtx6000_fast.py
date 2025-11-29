#!/usr/bin/env python3
"""
RTX 6000 Pro 96GB - FAST Training (<10 hours)
Optimized for speed while maintaining good quality.
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Optimizations
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from accelerate.utils import set_seed

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.data_processor import DataProcessor


@dataclass
class RTX6000FastConfig:
    """RTX 6000 Pro 96GB - Optimized for <10 hour training."""

    # Model
    model_name: str = "Qwen/Qwen3-4B"

    # QLoRA settings - balanced
    use_qlora: bool = True
    lora_r: int = 32  # Smaller rank = faster
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # Precision
    precision: str = "4bit"

    # Batch settings - optimized for RTX 6000 Pro
    batch_size: int = 32  # Good for 96GB
    gradient_accumulation_steps: int = 2  # Effective batch = 64
    max_length: int = 384  # Shorter = faster

    # Training - FAST
    num_epochs: int = 8  # Fewer epochs
    learning_rate: float = 3e-4  # Higher LR for faster convergence
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03  # Less warmup
    max_grad_norm: float = 1.0

    # Dataset - smaller for speed
    max_examples: int = 25000  # 25K examples
    train_split: float = 0.95

    # Checkpointing
    output_dir: str = "./checkpoints_rtx6000"
    save_every_epochs: int = 2
    save_best: bool = True

    # Advanced
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    num_workers: int = 4

    # Seed
    seed: int = 42


class FastDataset(Dataset):
    """Pre-tokenized dataset for speed."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int):
        self.tokenized = []
        print("Pre-tokenizing dataset...")
        for ex in tqdm(examples, desc="Tokenizing"):
            full_text = f"{ex['input']} {ex['output']}"
            tokens = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            labels = tokens["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            self.tokenized.append({
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0)
            })

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        return self.tokenized[idx]


class FastModel(nn.Module):
    """QLoRA model optimized for speed."""

    def __init__(self, config: RTX6000FastConfig):
        super().__init__()
        self.config = config

        print(f"\nLoading {config.model_name} with QLoRA (FAST mode)...")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<bot>", "<eot>"]})
        self.bot_token_id = self.tokenizer.convert_tokens_to_ids("<bot>")
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<eot>")
        print(f"Special tokens: <bot>={self.bot_token_id}, <eot>={self.eot_token_id}")

        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2" if config.flash_attention else "sdpa",
            use_cache=False
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = prepare_model_for_kbit_training(self.model)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.lora_target_modules),
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

        # Stats
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel ready!")
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"  LoRA r={config.lora_r}, alpha={config.lora_alpha}")

    def forward(self, input_ids, attention_mask, labels=None):
        if labels is not None:
            labels = labels.clone()
            labels[input_ids == self.bot_token_id] = -100
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class FastTrainer:
    """Optimized trainer for speed."""

    def __init__(self, model: FastModel, config: RTX6000FastConfig, train_data, eval_data=None):
        self.model = model
        self.config = config

        self.accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )

        # Datasets
        self.train_dataset = FastDataset(train_data, model.tokenizer, config.max_length)
        self.eval_dataset = FastDataset(eval_data, model.tokenizer, config.max_length) if eval_data else None

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        ) if self.eval_dataset else None

        # Calculate steps
        self.steps_per_epoch = len(self.train_loader) // config.gradient_accumulation_steps
        self.total_steps = self.steps_per_epoch * config.num_epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)

        print(f"\nTraining setup:")
        print(f"  Train: {len(train_data)}, Eval: {len(eval_data) if eval_data else 0}")
        print(f"  Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
        print(f"  Steps/epoch: {self.steps_per_epoch}, Total: {self.total_steps}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )

        # Prepare
        self.model.model, self.optimizer, self.train_loader, self.scheduler = \
            self.accelerator.prepare(self.model.model, self.optimizer, self.train_loader, self.scheduler)

        if self.eval_loader:
            self.eval_loader = self.accelerator.prepare(self.eval_loader)

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.best_loss = float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.accelerator.is_local_main_process)

        for batch_idx, batch in enumerate(pbar):
            with self.accelerator.accumulate(self.model.model):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

                loss = outputs.loss
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                num_batches += 1

                if self.accelerator.sync_gradients:
                    self.global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{total_loss/num_batches:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

        return total_loss / num_batches

    def evaluate(self):
        if not self.eval_loader:
            return 0.0
        self.model.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch in self.eval_loader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                total_loss += outputs.loss.item()
                num_batches += 1
        return total_loss / num_batches

    def train(self):
        print("\n" + "=" * 60)
        print("   RTX 6000 PRO FAST TRAINING (<10 hours)")
        print("=" * 60)

        start_time = datetime.now()

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            eval_loss = self.evaluate() if self.eval_loader else 0

            elapsed = datetime.now() - start_time
            remaining = elapsed / epoch * (self.config.num_epochs - epoch)

            print(f"\nEpoch {epoch}/{self.config.num_epochs} | Train: {train_loss:.4f} | Eval: {eval_loss:.4f}")
            print(f"  Time: {elapsed} elapsed, ~{remaining} remaining")

            if eval_loss and eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.save("best")

            if epoch % self.config.save_every_epochs == 0:
                self.save(f"epoch_{epoch}")

        self.save("final")
        print(f"\n{'='*60}")
        print(f"DONE! Total time: {datetime.now() - start_time}")
        print(f"Model saved to: {self.config.output_dir}")
        print("=" * 60)

    def save(self, name):
        if not self.accelerator.is_local_main_process:
            return
        path = Path(self.config.output_dir) / name
        unwrapped = self.accelerator.unwrap_model(self.model.model)
        unwrapped.save_pretrained(path)
        self.model.tokenizer.save_pretrained(path)
        print(f"  Saved: {path}")


def main():
    config = RTX6000FastConfig()

    print("\n" + "=" * 60)
    print("   RTX 6000 PRO - FAST TRAINING (<10 hours)")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"QLoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Dataset: {config.max_examples} examples")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    set_seed(config.seed)

    # Load model
    model = FastModel(config)

    # Load data
    print("\nLoading dataset...")
    processor = DataProcessor(cache_dir="./data_cache")
    examples = processor.load_all_datasets()

    # Limit dataset
    if len(examples) > config.max_examples:
        print(f"Limiting: {len(examples)} → {config.max_examples}")
        random.shuffle(examples)
        examples = examples[:config.max_examples]

    # Create adaptive data
    adaptive_data = processor.create_adaptive_dataset(examples, max_latent_tokens=15)

    # Split
    random.shuffle(adaptive_data)
    split_idx = int(len(adaptive_data) * config.train_split)
    train_data = adaptive_data[:split_idx]
    eval_data = adaptive_data[split_idx:]

    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Train
    trainer = FastTrainer(model, config, train_data, eval_data)
    trainer.train()


if __name__ == "__main__":
    main()
