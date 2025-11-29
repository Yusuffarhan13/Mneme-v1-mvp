#!/usr/bin/env python3
"""
Coconut Training Script
Multi-stage curriculum training for latent space reasoning.

Usage:
    python train_coconut.py                    # Train with defaults
    python train_coconut.py --config config.yaml
    python train_coconut.py --stage 0          # Train specific stage
    python train_coconut.py --adaptive         # Adaptive training (mixed stages)
"""

import os
import sys
import yaml
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_processor import DataProcessor, CoconutExample
from training.coconut_model import CoconutQwen


# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-4B"
    max_length: int = 512

    # QLoRA
    use_qlora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Training
    num_stages: int = 5
    epochs_per_stage: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0

    # Data
    cache_dir: str = "./data_cache"
    train_split: float = 0.9
    seed: int = 42

    # Checkpoints
    output_dir: str = "./checkpoints"
    save_every_stage: bool = True
    resume_from: Optional[str] = None

    # Logging
    log_steps: int = 50
    eval_steps: int = 500

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            model_name=data.get("model", {}).get("name", cls.model_name),
            max_length=data.get("model", {}).get("max_length", cls.max_length),
            use_qlora=data.get("qlora", {}).get("enabled", cls.use_qlora),
            lora_r=data.get("qlora", {}).get("r", cls.lora_r),
            lora_alpha=data.get("qlora", {}).get("alpha", cls.lora_alpha),
            lora_dropout=data.get("qlora", {}).get("dropout", cls.lora_dropout),
            num_stages=data.get("training", {}).get("num_stages", cls.num_stages),
            epochs_per_stage=data.get("training", {}).get("epochs_per_stage", cls.epochs_per_stage),
            learning_rate=data.get("training", {}).get("learning_rate", cls.learning_rate),
            weight_decay=data.get("training", {}).get("weight_decay", cls.weight_decay),
            warmup_ratio=data.get("training", {}).get("warmup_ratio", cls.warmup_ratio),
            batch_size=data.get("training", {}).get("batch_size", cls.batch_size),
            gradient_accumulation_steps=data.get("training", {}).get("gradient_accumulation_steps", cls.gradient_accumulation_steps),
            max_grad_norm=data.get("training", {}).get("max_grad_norm", cls.max_grad_norm),
            cache_dir=data.get("data", {}).get("cache_dir", cls.cache_dir),
            train_split=data.get("data", {}).get("train_split", cls.train_split),
            seed=data.get("data", {}).get("seed", cls.seed),
            output_dir=data.get("checkpointing", {}).get("output_dir", cls.output_dir),
        )


class CoconutDataset(Dataset):
    """Dataset for Coconut training."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Format: input + output
        full_text = f"{example['input']} {example['output']}"

        # Tokenize
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Create labels
        labels = tokens["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


class CoconutTrainer:
    """
    Multi-stage curriculum trainer for Coconut.
    """

    def __init__(
        self,
        model: CoconutQwen,
        config: TrainingConfig,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None
    ):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.eval_data = eval_data

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Training state
        self.global_step = 0
        self.current_stage = 0
        self.best_loss = float("inf")

    def create_dataloader(self, data: List[Dict], shuffle: bool = True) -> DataLoader:
        """Create DataLoader from data."""
        dataset = CoconutDataset(
            examples=data,
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_length
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True
        )

    def train_epoch(self, dataloader: DataLoader, stage: int, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Stage {stage} Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / self.config.gradient_accumulation_steps
            loss.backward()

            total_loss += outputs.loss.item()
            num_batches += 1

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Update progress
            avg_loss = total_loss / num_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": self.global_step})

        return total_loss / num_batches

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train_stage(self, stage: int, stage_data: List[Dict]) -> float:
        """Train a single curriculum stage."""
        print(f"\n{'='*60}")
        print(f"Training Stage {stage}")
        print(f"{'='*60}")
        print(f"Examples: {len(stage_data)}")

        # Create dataloaders
        train_loader = self.create_dataloader(stage_data)
        eval_loader = None
        if self.eval_data:
            eval_loader = self.create_dataloader(self.eval_data, shuffle=False)

        best_stage_loss = float("inf")

        for epoch in range(self.config.epochs_per_stage):
            # Train
            train_loss = self.train_epoch(train_loader, stage, epoch + 1)
            print(f"Stage {stage} Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

            # Evaluate
            if eval_loader:
                eval_loss = self.evaluate(eval_loader)
                print(f"Stage {stage} Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")

                if eval_loss < best_stage_loss:
                    best_stage_loss = eval_loss
            else:
                best_stage_loss = train_loss

        return best_stage_loss

    def train_curriculum(self, curriculum: Dict[int, List[Dict]]):
        """
        Train through all curriculum stages.

        Args:
            curriculum: Dict mapping stage number to training examples
        """
        print("\n" + "=" * 60)
        print("Starting Coconut Curriculum Training")
        print("=" * 60)
        print(f"Total stages: {len(curriculum)}")
        print(f"Epochs per stage: {self.config.epochs_per_stage}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print("=" * 60)

        stage_losses = {}

        for stage in sorted(curriculum.keys()):
            self.current_stage = stage
            stage_data = curriculum[stage]

            # Train stage
            stage_loss = self.train_stage(stage, stage_data)
            stage_losses[stage] = stage_loss

            # Save checkpoint
            if self.config.save_every_stage:
                self.save_checkpoint(f"stage_{stage}")

            print(f"\nStage {stage} complete - Best Loss: {stage_loss:.4f}\n")

        # Final save
        self.save_checkpoint("final")

        # Summary
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        for stage, loss in stage_losses.items():
            print(f"Stage {stage}: {loss:.4f}")
        print("=" * 60)

        return stage_losses

    def train_adaptive(self, adaptive_data: List[Dict], num_epochs: int = 10):
        """
        Train with adaptive/mixed data (all stages combined).

        Args:
            adaptive_data: Mixed training data with variable latent lengths
            num_epochs: Number of epochs to train
        """
        print("\n" + "=" * 60)
        print("Starting Adaptive Coconut Training")
        print("=" * 60)
        print(f"Examples: {len(adaptive_data)}")
        print(f"Epochs: {num_epochs}")
        print("=" * 60)

        train_loader = self.create_dataloader(adaptive_data)
        eval_loader = None
        if self.eval_data:
            eval_loader = self.create_dataloader(self.eval_data, shuffle=False)

        best_loss = float("inf")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, stage="adaptive", epoch=epoch + 1)
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

            # Evaluate
            if eval_loader:
                eval_loss = self.evaluate(eval_loader)
                print(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.save_checkpoint("best")
            else:
                if train_loss < best_loss:
                    best_loss = train_loss
                    self.save_checkpoint("best")

            # Save periodic checkpoint
            if (epoch + 1) % 3 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}")

        # Final save
        self.save_checkpoint("final")

        print("\n" + "=" * 60)
        print(f"Adaptive Training Complete! Best Loss: {best_loss:.4f}")
        print("=" * 60)

        return best_loss

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = Path(self.config.output_dir) / name
        self.model.save_pretrained(str(path))

        # Save training state
        state = {
            "global_step": self.global_step,
            "current_stage": self.current_stage,
            "best_loss": self.best_loss,
            "config": vars(self.config)
        }
        with open(path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Coconut Training")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--stage", type=int, default=None,
                       help="Train specific stage only")
    parser.add_argument("--adaptive", action="store_true",
                       help="Use adaptive training (mixed stages)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Epochs for adaptive training")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="./checkpoints",
                       help="Output directory")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    if config_path.exists():
        config = TrainingConfig.from_yaml(str(config_path))
    else:
        config = TrainingConfig()

    if args.output:
        config.output_dir = args.output

    # Set seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("\n" + "=" * 60)
    print("Coconut Training for Latent Space Reasoning")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"QLoRA: {config.use_qlora}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = CoconutQwen(
        model_id=config.model_name,
        use_qlora=config.use_qlora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )

    # Load data
    print("\nLoading datasets...")
    processor = DataProcessor(cache_dir=config.cache_dir)
    examples = processor.load_all_datasets()

    # Limit dataset size for practical training time
    max_examples = 10000  # ~2-3 hours per epoch
    if len(examples) > max_examples:
        print(f"Limiting dataset from {len(examples)} to {max_examples} examples")
        random.shuffle(examples)
        examples = examples[:max_examples]

    # Split train/eval
    random.shuffle(examples)
    split_idx = int(len(examples) * config.train_split)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]

    print(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Create eval data
    eval_data = None
    if eval_examples:
        eval_data = [processor.create_stage_example(e, stage=0) for e in eval_examples]

    # Create trainer
    trainer = CoconutTrainer(
        model=model,
        config=config,
        train_data=[],  # Will be set per stage
        eval_data=eval_data
    )

    if args.adaptive:
        # Adaptive training with mixed stages
        adaptive_data = processor.create_adaptive_dataset(train_examples)
        trainer.train_adaptive(adaptive_data, num_epochs=args.epochs)
    elif args.stage is not None:
        # Train specific stage
        stage_data = [processor.create_stage_example(e, args.stage) for e in train_examples]
        trainer.train_stage(args.stage, stage_data)
        trainer.save_checkpoint(f"stage_{args.stage}")
    else:
        # Full curriculum training
        curriculum = processor.create_curriculum_dataset(
            train_examples,
            num_stages=config.num_stages
        )
        trainer.train_curriculum(curriculum)

    print("\nDone!")


if __name__ == "__main__":
    main()
