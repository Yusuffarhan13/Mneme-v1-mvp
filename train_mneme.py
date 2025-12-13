"""
Training Script for Mneme Memory Encoder

Trains the hypernetwork to produce weight deltas that encode facts.

Training Objective:
1. Inject fact via weight deltas
2. Ask question about the fact
3. Model should answer correctly
4. Backprop through encoder

This teaches the encoder WHAT weight modifications encode facts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import os
import random
from tqdm import tqdm

from mneme import MnemeModel, MnemeConfig, MemoryEncoder


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-4B"

    # Training
    learning_rate: float = 2e-4  # Slightly higher LR
    batch_size: int = 1
    num_epochs: int = 10  # More epochs
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Memory
    delta_rank: int = 4
    target_layers: List[int] = None

    # Output
    output_dir: str = "mneme_trained"

    def __post_init__(self):
        if self.target_layers is None:
            self.target_layers = [8, 16, 24]


class FactQADataset(Dataset):
    """
    Dataset of (fact, question, answer) triplets.

    The encoder learns to produce weight deltas such that:
    - When delta is applied
    - And question is asked
    - Model produces correct answer
    """

    def __init__(self, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._create_data()

    def _create_data(self) -> List[Dict]:
        """Create training data."""
        data = []

        # Pattern: (fact_template, question_template, answer_template)
        patterns = [
            # Names
            ("The user's name is {name}.", "What is the user's name?", "The user's name is {name}."),
            ("The user is called {name}.", "What is the user called?", "The user is called {name}."),

            # Locations
            ("The user lives in {place}.", "Where does the user live?", "The user lives in {place}."),
            ("The user is from {place}.", "Where is the user from?", "The user is from {place}."),

            # Preferences
            ("The user's favorite color is {color}.", "What is the user's favorite color?", "The user's favorite color is {color}."),
            ("The user likes {thing}.", "What does the user like?", "The user likes {thing}."),
            ("The user loves {thing}.", "What does the user love?", "The user loves {thing}."),

            # Jobs
            ("The user works as a {job}.", "What does the user work as?", "The user works as a {job}."),
            ("The user is a {job}.", "What is the user's job?", "The user is a {job}."),

            # Workplaces
            ("The user works at {company}.", "Where does the user work?", "The user works at {company}."),
            ("The user is employed at {company}.", "Where is the user employed?", "The user is employed at {company}."),

            # Age
            ("The user is {age} years old.", "How old is the user?", "The user is {age} years old."),

            # Food
            ("The user's favorite food is {food}.", "What is the user's favorite food?", "The user's favorite food is {food}."),

            # Pets
            ("The user has a {pet} named {petname}.", "What pet does the user have?", "The user has a {pet} named {petname}."),

            # Numbers/Facts
            ("The password is {password}.", "What is the password?", "The password is {password}."),
            ("The code is {code}.", "What is the code?", "The code is {code}."),
            ("The meeting is at {time}.", "When is the meeting?", "The meeting is at {time}."),
        ]

        # Fill in values
        names = ["Alice", "Bob", "Yusuf", "Sarah", "Mike", "Emma", "John", "Lisa", "David", "Maria"]
        places = ["Dubai", "Tokyo", "London", "Paris", "New York", "Sydney", "Berlin", "Toronto", "Mumbai", "Seoul"]
        colors = ["blue", "red", "green", "purple", "orange", "yellow", "black", "white", "pink", "gold"]
        things = ["programming", "music", "reading", "gaming", "cooking", "hiking", "art", "movies", "travel", "sports"]
        jobs = ["software engineer", "doctor", "teacher", "designer", "writer", "chef", "artist", "lawyer", "nurse", "scientist"]
        companies = ["Google", "Microsoft", "Apple", "Amazon", "Meta", "Netflix", "Tesla", "OpenAI", "Anthropic", "NASA"]
        ages = ["25", "30", "28", "35", "22", "40", "32", "27", "45", "38"]
        foods = ["pizza", "sushi", "pasta", "tacos", "curry", "ramen", "steak", "salad", "burger", "seafood"]
        pets = ["dog", "cat", "bird", "fish", "hamster", "rabbit", "turtle", "parrot", "snake", "lizard"]
        petnames = ["Max", "Luna", "Charlie", "Bella", "Rocky", "Milo", "Coco", "Buddy", "Daisy", "Oscar"]
        passwords = ["secret123", "mypass456", "hello789", "test2024", "admin001"]
        codes = ["1234", "5678", "9012", "4321", "8765"]
        times = ["3pm", "10am", "2:30pm", "9am", "5pm", "noon", "4pm", "11am"]

        # Generate examples
        for fact_t, q_t, a_t in patterns:
            for _ in range(10):  # 10 examples per pattern for better coverage
                if "{name}" in fact_t:
                    name = random.choice(names)
                    fact = fact_t.format(name=name)
                    question = q_t
                    answer = a_t.format(name=name)
                elif "{place}" in fact_t:
                    place = random.choice(places)
                    fact = fact_t.format(place=place)
                    question = q_t
                    answer = a_t.format(place=place)
                elif "{color}" in fact_t:
                    color = random.choice(colors)
                    fact = fact_t.format(color=color)
                    question = q_t
                    answer = a_t.format(color=color)
                elif "{thing}" in fact_t:
                    thing = random.choice(things)
                    fact = fact_t.format(thing=thing)
                    question = q_t
                    answer = a_t.format(thing=thing)
                elif "{job}" in fact_t:
                    job = random.choice(jobs)
                    fact = fact_t.format(job=job)
                    question = q_t
                    answer = a_t.format(job=job)
                elif "{company}" in fact_t:
                    company = random.choice(companies)
                    fact = fact_t.format(company=company)
                    question = q_t
                    answer = a_t.format(company=company)
                elif "{age}" in fact_t:
                    age = random.choice(ages)
                    fact = fact_t.format(age=age)
                    question = q_t
                    answer = a_t.format(age=age)
                elif "{food}" in fact_t:
                    food = random.choice(foods)
                    fact = fact_t.format(food=food)
                    question = q_t
                    answer = a_t.format(food=food)
                elif "{pet}" in fact_t:
                    pet = random.choice(pets)
                    petname = random.choice(petnames)
                    fact = fact_t.format(pet=pet, petname=petname)
                    question = q_t
                    answer = a_t.format(pet=pet, petname=petname)
                elif "{password}" in fact_t:
                    password = random.choice(passwords)
                    fact = fact_t.format(password=password)
                    question = q_t
                    answer = a_t.format(password=password)
                elif "{code}" in fact_t:
                    code = random.choice(codes)
                    fact = fact_t.format(code=code)
                    question = q_t
                    answer = a_t.format(code=code)
                elif "{time}" in fact_t:
                    time = random.choice(times)
                    fact = fact_t.format(time=time)
                    question = q_t
                    answer = a_t.format(time=time)
                else:
                    continue

                data.append({
                    "fact": fact,
                    "question": question,
                    "answer": answer
                })

        random.shuffle(data)
        print(f"Created {len(data)} training examples")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        return self.data[idx]


class MnemeTrainer:
    """
    Trains the Memory Encoder to produce effective weight deltas.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )

        # Freeze base model completely
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        print("Creating Mneme model...")
        mneme_config = MnemeConfig(
            delta_rank=config.delta_rank,
            target_layers=config.target_layers,
        )
        self.model = MnemeModel(self.base_model, self.tokenizer, mneme_config)

        # Only train the encoder
        self.trainable_params = list(self.model.encoder.parameters())
        num_params = sum(p.numel() for p in self.trainable_params)
        print(f"Trainable parameters: {num_params:,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01
        )

        self.global_step = 0
        self.best_loss = float('inf')

    def compute_loss(self, fact: str, question: str, answer: str) -> torch.Tensor:
        """
        Compute loss for a single (fact, question, answer) triplet.

        Uses forward hooks to apply deltas during forward pass (preserves gradients).
        """
        # Tokenize fact
        fact_inputs = self.tokenizer(
            fact,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=True
        )
        fact_inputs = {k: v.to(self.device) for k, v in fact_inputs.items()}

        # Get weight deltas from encoder (with gradient)
        deltas = self.model.encoder(fact_inputs["input_ids"], fact_inputs.get("attention_mask"))

        # Create hooks to apply deltas during forward pass
        hooks = []
        scaling = 1.0 / self.model.config.delta_rank

        for layer_idx in self.model.config.target_layers:
            layer = self.base_model.model.layers[layer_idx]

            # Get deltas for this layer
            gate_A = deltas.get(f"layer_{layer_idx}_gate_A")
            gate_B = deltas.get(f"layer_{layer_idx}_gate_B")
            up_A = deltas.get(f"layer_{layer_idx}_up_A")
            up_B = deltas.get(f"layer_{layer_idx}_up_B")
            down_A = deltas.get(f"layer_{layer_idx}_down_A")
            down_B = deltas.get(f"layer_{layer_idx}_down_B")

            # Create hook for gate_proj
            if gate_A is not None and gate_B is not None:
                gA = gate_A.squeeze(0).to(self.base_model.dtype)
                gB = gate_B.squeeze(0).to(self.base_model.dtype)
                def make_gate_hook(A, B):
                    def hook(module, input, output):
                        # output = input @ W.T, we want output + input @ delta.T
                        # delta = A @ B.T, so delta.T = B @ A.T
                        # input @ delta.T = input @ B @ A.T
                        inp = input[0] if isinstance(input, tuple) else input
                        delta_out = inp @ B @ A.T * scaling
                        return output + delta_out
                    return hook
                h = layer.mlp.gate_proj.register_forward_hook(make_gate_hook(gA, gB))
                hooks.append(h)

            # Create hook for up_proj
            if up_A is not None and up_B is not None:
                uA = up_A.squeeze(0).to(self.base_model.dtype)
                uB = up_B.squeeze(0).to(self.base_model.dtype)
                def make_up_hook(A, B):
                    def hook(module, input, output):
                        inp = input[0] if isinstance(input, tuple) else input
                        delta_out = inp @ B @ A.T * scaling
                        return output + delta_out
                    return hook
                h = layer.mlp.up_proj.register_forward_hook(make_up_hook(uA, uB))
                hooks.append(h)

            # Create hook for down_proj
            if down_A is not None and down_B is not None:
                dA = down_A.squeeze(0).to(self.base_model.dtype)
                dB = down_B.squeeze(0).to(self.base_model.dtype)
                def make_down_hook(A, B):
                    def hook(module, input, output):
                        inp = input[0] if isinstance(input, tuple) else input
                        delta_out = inp @ B @ A.T * scaling
                        return output + delta_out
                    return hook
                h = layer.mlp.down_proj.register_forward_hook(make_down_hook(dA, dB))
                hooks.append(h)

        try:
            # Create QA prompt
            qa_text = f"Question: {question}\nAnswer: {answer}"

            qa_inputs = self.tokenizer(
                qa_text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            qa_inputs = {k: v.to(self.device) for k, v in qa_inputs.items()}

            # Compute language modeling loss
            labels = qa_inputs["input_ids"].clone()

            # Only compute loss on answer part
            answer_start = qa_text.find("Answer:")
            if answer_start > 0:
                answer_token_start = len(self.tokenizer.encode(qa_text[:answer_start], add_special_tokens=False))
                labels[0, :answer_token_start] = -100

            outputs = self.base_model(
                input_ids=qa_inputs["input_ids"],
                attention_mask=qa_inputs["attention_mask"],
                labels=labels,
            )

            return outputs.loss

        finally:
            # Remove all hooks
            for h in hooks:
                h.remove()

    def train_step(self, batch: Dict) -> float:
        """Single training step."""
        self.model.encoder.train()

        fact = batch["fact"]
        question = batch["question"]
        answer = batch["answer"]

        loss = self.compute_loss(fact, question, answer)

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def train(self):
        """Main training loop."""
        # Create dataset
        dataset = FactQADataset(self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x[0]  # Return single item
        )

        os.makedirs(self.config.output_dir, exist_ok=True)

        print(f"Training for {self.config.num_epochs} epochs...")

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                try:
                    loss = self.train_step(batch)
                    epoch_loss += loss
                    num_batches += 1

                    # Gradient accumulation
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.trainable_params,
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1

                    # Update progress
                    avg_loss = epoch_loss / max(num_batches, 1)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    self.optimizer.zero_grad()
                    self.model._restore_weights()
                    continue

            # End of epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

            # Save best model
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.save_checkpoint("best_encoder.pt")

        # Save final
        self.save_checkpoint("final_encoder.pt")
        print("Training complete!")

    def save_checkpoint(self, filename: str):
        """Save encoder checkpoint."""
        path = os.path.join(self.config.output_dir, filename)
        torch.save({
            "encoder_state": self.model.encoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": {
                "delta_rank": self.config.delta_rank,
                "target_layers": self.config.target_layers,
            }
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load encoder checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint["encoder_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        print(f"Loaded checkpoint from {path}")


def evaluate_memory(model: MnemeModel, facts_and_questions: List[Tuple[str, str, str]]):
    """
    Evaluate memory injection.

    Args:
        model: MnemeModel
        facts_and_questions: List of (fact, question, expected_answer)
    """
    print("\n=== Memory Evaluation ===")

    model.clear_memories()

    for fact, question, expected in facts_and_questions:
        print(f"\nFact: {fact}")

        # Inject memory
        mem_id = model.inject_memory(fact)
        print(f"Injected as: {mem_id}")

        # Ask question
        prompt = f"Question: {question}\nAnswer:"
        inputs = model.tokenizer(prompt, return_tensors="pt").to(model.model_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
            )

        response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip()

        print(f"Question: {question}")
        print(f"Expected: {expected}")
        print(f"Got: {answer}")
        print(f"Match: {'✓' if expected.lower() in answer.lower() else '✗'}")

    model.clear_memories()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Mneme Memory Encoder")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=4, help="Delta rank")
    parser.add_argument("--output", type=str, default="mneme_trained", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        delta_rank=args.rank,
        output_dir=args.output,
    )

    trainer = MnemeTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()
