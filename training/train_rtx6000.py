#!/usr/bin/env python3
"""
RTX 6000 Pro (96GB) Optimized Full Fine-Tuning for Coconut
Maximum settings - trains ALL parameters on full dataset.

Usage:
    python training/train_h100.py
"""

import os
import sys
import json
import random
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# RTX 6000 Pro Optimizations
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # Ada Lovelace architecture
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid fork warnings

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
from accelerate.utils import set_seed

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.data_processor import DataProcessor


@dataclass
class RTX6000Config:
    """Maxed out config for RTX 6000 Pro 96GB training."""

    # Model
    model_name: str = "Qwen/Qwen3-4B"

    # Full fine-tuning (NO QLoRA - train everything)
    full_finetune: bool = True

    # Precision - bf16 for efficiency
    precision: str = "bf16"

    # Batch settings - optimized for full fine-tuning
    batch_size: int = 16  # Safe batch size for full fine-tuning
    gradient_accumulation_steps: int = 8  # Effective batch = 128
    max_length: int = 512  # Standard context length

    # Training
    num_epochs: int = 20  # Balanced training time
    learning_rate: float = 2e-5  # Full fine-tune LR
    min_learning_rate: float = 1e-6  # Minimum LR for cosine
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05  # 5% warmup
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_type: str = "cosine"  # cosine, linear, cosine_restarts

    # Dataset - NO LIMITS
    use_full_dataset: bool = True
    train_split: float = 0.95  # 95% train, 5% eval

    # Checkpointing
    output_dir: str = "./checkpoints_rtx6000"
    save_every_epochs: int = 5
    save_best: bool = True

    # Logging
    log_steps: int = 10
    eval_steps: int = 500

    # Advanced
    gradient_checkpointing: bool = True  # Enable for memory efficiency
    flash_attention: bool = True  # RTX 6000 supports flash attention
    compile_model: bool = False  # Disable - causes issues with flash attention

    # Seed
    seed: int = 42


class CoconutDatasetH100(Dataset):
    """Optimized dataset for H100 training."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 768):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-tokenize for speed
        print("Pre-tokenizing dataset...")
        self.tokenized = []
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


class CoconutModelRTX6000(nn.Module):
    """Full fine-tuning model wrapper for RTX 6000 Pro 96GB."""

    BOT_TOKEN = "<bot>"
    EOT_TOKEN = "<eot>"

    def __init__(self, config: RTX6000Config):
        super().__init__()
        self.config = config

        print(f"\nLoading {config.model_name} for FULL fine-tuning...")
        print("All parameters will be trained!")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._add_special_tokens()

        # Load model in bf16 (H100 native)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2" if config.flash_attention else "sdpa",
            use_cache=False  # Disable for training
        )

        # Resize embeddings for special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Enable gradient checkpointing
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Unfreeze ALL parameters
        for param in self.model.parameters():
            param.requires_grad = True

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\nModel loaded!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} (100%)")
        print(f"  Precision: {config.precision}")
        print(f"  Flash Attention: {config.flash_attention}")

    def _add_special_tokens(self):
        """Add <bot> and <eot> tokens."""
        special_tokens = {"additional_special_tokens": [self.BOT_TOKEN, self.EOT_TOKEN]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)

        self.bot_token_id = self.tokenizer.convert_tokens_to_ids(self.BOT_TOKEN)
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids(self.EOT_TOKEN)

        print(f"Added special tokens: <bot>={self.bot_token_id}, <eot>={self.eot_token_id}")

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass with loss masking on <bot> tokens."""

        # Mask loss on <bot> tokens
        if labels is not None:
            labels = labels.clone()
            bot_mask = (input_ids == self.bot_token_id)
            labels[bot_mask] = -100

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs

    def save(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class RTX6000Trainer:
    """Optimized trainer for RTX 6000 Pro 96GB."""

    def __init__(
        self,
        model: CoconutModelRTX6000,
        config: RTX6000Config,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None
    ):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.eval_data = eval_data

        # Setup accelerator for distributed training if available
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )

        # Create datasets
        self.train_dataset = CoconutDatasetH100(
            train_data, model.tokenizer, config.max_length
        )
        self.eval_dataset = None
        if eval_data:
            self.eval_dataset = CoconutDatasetH100(
                eval_data, model.tokenizer, config.max_length
            )

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        self.eval_loader = None
        if self.eval_dataset:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=config.batch_size * 2,  # Larger for eval
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

        # Calculate training steps
        self.steps_per_epoch = len(self.train_loader) // config.gradient_accumulation_steps
        self.total_steps = self.steps_per_epoch * config.num_epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)

        print(f"\nTraining setup:")
        print(f"  Train examples: {len(train_data)}")
        print(f"  Eval examples: {len(eval_data) if eval_data else 0}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"  Steps per epoch: {self.steps_per_epoch}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Warmup steps: {self.warmup_steps}")

        # Optimizer - AdamW with proper settings
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
            eps=1e-8
        )

        # Scheduler
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps
            )
        elif config.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps
            )
        else:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.steps_per_epoch,
                T_mult=2,
                eta_min=config.min_learning_rate
            )

        # Prepare with accelerator
        self.model.model, self.optimizer, self.train_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model.model, self.optimizer, self.train_loader, self.scheduler
            )

        if self.eval_loader:
            self.eval_loader = self.accelerator.prepare(self.eval_loader)

        # Compile model for speed (H100 benefits greatly)
        if config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            try:
                self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
                print("Model compiled successfully!")
            except Exception as e:
                print(f"Could not compile model: {e}")

        # Output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process
        )

        for batch_idx, batch in enumerate(pbar):
            with self.accelerator.accumulate(self.model.model):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

                loss = outputs.loss
                self.accelerator.backward(loss)

                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.config.max_grad_norm
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                num_batches += 1

                if self.accelerator.sync_gradients:
                    self.global_step += 1

                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/num_batches:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": self.global_step
                })

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    self.accelerator.log({
                        "train_loss": loss.item(),
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "step": self.global_step
                    })

        return total_loss / num_batches

    def evaluate(self) -> float:
        """Evaluate model."""
        if not self.eval_loader:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                total_loss += outputs.loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self):
        """Full training loop."""
        print("\n" + "=" * 70)
        print("   H100 FULL FINE-TUNING - COCONUT METHOD")
        print("=" * 70)
        print(f"Training {self.config.model_name}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Total steps: {self.total_steps}")
        print("=" * 70 + "\n")

        start_time = datetime.now()

        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Evaluate
            eval_loss = self.evaluate() if self.eval_loader else 0

            # Log
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if eval_loss:
                print(f"  Eval Loss: {eval_loss:.4f}")

            # Save best model
            if self.config.save_best and eval_loss and eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.save_checkpoint("best")
                print(f"  New best model saved! (loss: {eval_loss:.4f})")

            # Save periodic checkpoint
            if epoch % self.config.save_every_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch}")

            # Time estimate
            elapsed = datetime.now() - start_time
            remaining = elapsed / epoch * (self.config.num_epochs - epoch)
            print(f"  Time: {elapsed} elapsed, ~{remaining} remaining")

        # Save final model
        self.save_checkpoint("final")

        total_time = datetime.now() - start_time
        print("\n" + "=" * 70)
        print("   TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Total time: {total_time}")
        print(f"Best eval loss: {self.best_loss:.4f}")
        print(f"Model saved to: {self.config.output_dir}")
        print("=" * 70)

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        if not self.accelerator.is_local_main_process:
            return

        path = Path(self.config.output_dir) / name

        # Unwrap model if needed
        unwrapped = self.accelerator.unwrap_model(self.model.model)
        unwrapped.save_pretrained(path)
        self.model.tokenizer.save_pretrained(path)

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": vars(self.config)
        }
        with open(path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"  Checkpoint saved: {path}")


def main():
    # Config
    config = RTX6000Config()

    print("\n" + "=" * 70)
    print("   COCONUT TRAINING - RTX 6000 PRO 96GB OPTIMIZED")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Full Fine-Tuning: {config.full_finetune} (ALL parameters)")
    print(f"Precision: {config.precision}")
    print(f"Batch Size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning Rate: {config.learning_rate}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)

    # Set seed
    set_seed(config.seed)

    # Load model
    model = CoconutModelRTX6000(config)

    # Load data
    print("\nLoading dataset...")
    processor = DataProcessor(cache_dir="./data_cache")
    examples = processor.load_all_datasets()

    # Limit dataset for practical training time
    # 50K examples = ~1.5 hours per epoch = ~2 days total
    MAX_EXAMPLES = 50000
    if len(examples) > MAX_EXAMPLES:
        print(f"Limiting dataset from {len(examples)} to {MAX_EXAMPLES} examples")
        random.shuffle(examples)
        examples = examples[:MAX_EXAMPLES]

    print(f"Total examples: {len(examples)}")

    # Create adaptive dataset
    adaptive_data = processor.create_adaptive_dataset(examples, max_latent_tokens=20)

    # Split
    random.shuffle(adaptive_data)
    split_idx = int(len(adaptive_data) * config.train_split)
    train_data = adaptive_data[:split_idx]
    eval_data = adaptive_data[split_idx:]

    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Create trainer
    trainer = RTX6000Trainer(
        model=model,
        config=config,
        train_data=train_data,
        eval_data=eval_data
    )

    # Train!
    trainer.train()

    print("\nDone! Model saved to:", config.output_dir)


if __name__ == "__main__":
    main()
