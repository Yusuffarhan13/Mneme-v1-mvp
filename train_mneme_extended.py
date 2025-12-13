"""
Extended Training Script for Mneme Memory Encoder

Comprehensive training for 60+ minutes with:
- Massive dataset (1000+ examples)
- 100 epochs with cosine LR scheduler
- Multiple question variants per fact
- Periodic evaluation
- Contrastive elements to differentiate facts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import random
import time
from tqdm import tqdm
from datetime import datetime

from mneme import MnemeModel, MnemeConfig, MemoryEncoder


@dataclass
class ExtendedTrainingConfig:
    """Extended training configuration."""
    model_name: str = "Qwen/Qwen3-4B"

    # Training - optimized for 60+ min training
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    batch_size: int = 1
    num_epochs: int = 100  # Many more epochs
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Memory
    delta_rank: int = 4
    target_layers: List[int] = None

    # Output
    output_dir: str = "mneme_trained"
    eval_every: int = 10  # Evaluate every N epochs
    save_every: int = 5   # Save checkpoint every N epochs

    def __post_init__(self):
        if self.target_layers is None:
            self.target_layers = [8, 16, 24]


class ExtendedFactQADataset(Dataset):
    """
    Extended dataset with 1000+ training examples.
    Multiple question variants per fact type.
    """

    def __init__(self, tokenizer, examples_per_pattern: int = 30):
        self.tokenizer = tokenizer
        self.examples_per_pattern = examples_per_pattern
        self.data = self._create_data()

    def _create_data(self) -> List[Dict]:
        """Create comprehensive training data."""
        data = []

        # Extended patterns with multiple question variants
        patterns = [
            # Names - multiple question styles
            ("The user's name is {name}.", [
                ("What is the user's name?", "The user's name is {name}."),
                ("What is my name?", "{name}."),
                ("Tell me the user's name.", "The user's name is {name}."),
                ("Who is the user?", "The user is {name}."),
            ]),
            ("My name is {name}.", [
                ("What is your name?", "My name is {name}."),
                ("What are you called?", "I am called {name}."),
                ("Tell me your name.", "My name is {name}."),
            ]),
            ("The person is called {name}.", [
                ("What is the person called?", "The person is called {name}."),
                ("What is the person's name?", "The person's name is {name}."),
            ]),

            # Workplaces - critical to get right
            ("The user works at {company}.", [
                ("Where does the user work?", "The user works at {company}."),
                ("What company does the user work for?", "The user works for {company}."),
                ("Where is the user employed?", "The user is employed at {company}."),
                ("Tell me the user's workplace.", "The user works at {company}."),
            ]),
            ("I work at {company}.", [
                ("Where do you work?", "I work at {company}."),
                ("What company do you work for?", "I work for {company}."),
                ("Where are you employed?", "I am employed at {company}."),
            ]),
            ("The user is employed by {company}.", [
                ("Who employs the user?", "The user is employed by {company}."),
                ("Where does the user work?", "The user works at {company}."),
            ]),

            # Pets - important for testing
            ("The user has a dog named {petname}.", [
                ("What is the user's dog's name?", "The user's dog is named {petname}."),
                ("What is the name of the user's dog?", "The dog's name is {petname}."),
                ("What is the user's pet called?", "The user's pet is called {petname}."),
                ("Tell me about the user's dog.", "The user has a dog named {petname}."),
            ]),
            ("The user has a cat named {petname}.", [
                ("What is the user's cat's name?", "The user's cat is named {petname}."),
                ("What is the name of the user's cat?", "The cat's name is {petname}."),
            ]),
            ("My dog is called {petname}.", [
                ("What is your dog called?", "My dog is called {petname}."),
                ("What is your dog's name?", "My dog's name is {petname}."),
            ]),

            # Locations
            ("The user lives in {place}.", [
                ("Where does the user live?", "The user lives in {place}."),
                ("What city does the user live in?", "The user lives in {place}."),
                ("Where is the user located?", "The user is located in {place}."),
            ]),
            ("I live in {place}.", [
                ("Where do you live?", "I live in {place}."),
                ("What is your city?", "I live in {place}."),
            ]),
            ("The user is from {place}.", [
                ("Where is the user from?", "The user is from {place}."),
                ("What is the user's hometown?", "The user is from {place}."),
            ]),

            # Colors
            ("The user's favorite color is {color}.", [
                ("What is the user's favorite color?", "The user's favorite color is {color}."),
                ("What color does the user like?", "The user likes {color}."),
            ]),
            ("My favorite color is {color}.", [
                ("What is your favorite color?", "My favorite color is {color}."),
                ("What color do you like best?", "I like {color} best."),
            ]),

            # Jobs
            ("The user works as a {job}.", [
                ("What does the user work as?", "The user works as a {job}."),
                ("What is the user's job?", "The user's job is {job}."),
                ("What is the user's profession?", "The user is a {job}."),
            ]),
            ("I am a {job}.", [
                ("What is your job?", "I am a {job}."),
                ("What do you do for work?", "I work as a {job}."),
            ]),

            # Age
            ("The user is {age} years old.", [
                ("How old is the user?", "The user is {age} years old."),
                ("What is the user's age?", "The user is {age}."),
            ]),
            ("I am {age} years old.", [
                ("How old are you?", "I am {age} years old."),
                ("What is your age?", "I am {age}."),
            ]),

            # Food
            ("The user's favorite food is {food}.", [
                ("What is the user's favorite food?", "The user's favorite food is {food}."),
                ("What food does the user like?", "The user likes {food}."),
            ]),
            ("My favorite food is {food}.", [
                ("What is your favorite food?", "My favorite food is {food}."),
                ("What do you like to eat?", "I like to eat {food}."),
            ]),

            # Things/hobbies
            ("The user likes {thing}.", [
                ("What does the user like?", "The user likes {thing}."),
                ("What are the user's hobbies?", "The user likes {thing}."),
            ]),
            ("The user loves {thing}.", [
                ("What does the user love?", "The user loves {thing}."),
                ("What is the user passionate about?", "The user loves {thing}."),
            ]),

            # Codes/passwords
            ("The password is {password}.", [
                ("What is the password?", "The password is {password}."),
                ("Tell me the password.", "The password is {password}."),
            ]),
            ("The code is {code}.", [
                ("What is the code?", "The code is {code}."),
                ("Tell me the code.", "The code is {code}."),
            ]),

            # Time/dates
            ("The meeting is at {time}.", [
                ("When is the meeting?", "The meeting is at {time}."),
                ("What time is the meeting?", "The meeting is at {time}."),
            ]),
            ("The birthday is on {date}.", [
                ("When is the birthday?", "The birthday is on {date}."),
                ("What is the birthday date?", "The birthday is on {date}."),
            ]),

            # Numbers
            ("The user's phone number is {phone}.", [
                ("What is the user's phone number?", "The user's phone number is {phone}."),
                ("Tell me the phone number.", "The phone number is {phone}."),
            ]),
            ("The address is {address}.", [
                ("What is the address?", "The address is {address}."),
                ("Tell me the address.", "The address is {address}."),
            ]),
        ]

        # Value pools
        names = ["Alice", "Bob", "Yusuf", "Sarah", "Mike", "Emma", "John", "Lisa",
                 "David", "Maria", "Alex", "Sophie", "James", "Emily", "Michael",
                 "Jessica", "Daniel", "Ashley", "Matthew", "Amanda", "Christopher",
                 "Jennifer", "Joshua", "Elizabeth", "Andrew", "Nicole", "Ryan", "Megan"]

        companies = ["Google", "Microsoft", "Apple", "Amazon", "Meta", "Netflix",
                     "Tesla", "OpenAI", "Anthropic", "NASA", "SpaceX", "Adobe",
                     "Salesforce", "Oracle", "IBM", "Intel", "Nvidia", "AMD",
                     "Uber", "Airbnb", "Twitter", "LinkedIn", "Spotify", "Stripe"]

        petnames = ["Max", "Luna", "Charlie", "Bella", "Rocky", "Milo", "Coco",
                    "Buddy", "Daisy", "Oscar", "Lucy", "Bear", "Duke", "Sadie",
                    "Tucker", "Bailey", "Cooper", "Stella", "Bentley", "Zoe"]

        places = ["Dubai", "Tokyo", "London", "Paris", "New York", "Sydney",
                  "Berlin", "Toronto", "Mumbai", "Seoul", "Singapore", "Hong Kong",
                  "San Francisco", "Los Angeles", "Chicago", "Boston", "Seattle",
                  "Miami", "Denver", "Austin", "Amsterdam", "Barcelona", "Rome"]

        colors = ["blue", "red", "green", "purple", "orange", "yellow", "black",
                  "white", "pink", "gold", "silver", "navy", "teal", "maroon"]

        jobs = ["software engineer", "doctor", "teacher", "designer", "writer",
                "chef", "artist", "lawyer", "nurse", "scientist", "architect",
                "photographer", "musician", "accountant", "pilot", "dentist"]

        ages = [str(i) for i in range(18, 70)]

        foods = ["pizza", "sushi", "pasta", "tacos", "curry", "ramen", "steak",
                 "salad", "burger", "seafood", "thai food", "chinese food",
                 "mexican food", "italian food", "indian food", "french food"]

        things = ["programming", "music", "reading", "gaming", "cooking", "hiking",
                  "art", "movies", "travel", "sports", "photography", "dancing",
                  "writing", "swimming", "yoga", "cycling", "gardening", "chess"]

        passwords = ["secret123", "mypass456", "hello789", "test2024", "admin001",
                     "qwerty12", "letmein99", "pass1234", "welcome1", "sunshine"]

        codes = ["1234", "5678", "9012", "4321", "8765", "2468", "1357", "9999",
                 "0000", "1111", "2222", "3333", "4444", "5555", "6666", "7777"]

        times = ["3pm", "10am", "2:30pm", "9am", "5pm", "noon", "4pm", "11am",
                 "8pm", "6:30pm", "7am", "1pm", "3:30pm", "midnight"]

        dates = ["January 5th", "March 15th", "July 4th", "December 25th",
                 "February 14th", "October 31st", "November 11th", "April 1st"]

        phones = ["555-1234", "555-5678", "555-9012", "555-3456", "555-7890"]

        addresses = ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm Way"]

        # Generate examples
        for fact_template, qa_pairs in patterns:
            for _ in range(self.examples_per_pattern):
                # Determine which placeholder to use
                filled_fact = fact_template

                if "{name}" in fact_template:
                    val = random.choice(names)
                    filled_fact = fact_template.format(name=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(name=val)
                        })

                elif "{company}" in fact_template:
                    val = random.choice(companies)
                    filled_fact = fact_template.format(company=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(company=val)
                        })

                elif "{petname}" in fact_template:
                    val = random.choice(petnames)
                    filled_fact = fact_template.format(petname=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(petname=val)
                        })

                elif "{place}" in fact_template:
                    val = random.choice(places)
                    filled_fact = fact_template.format(place=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(place=val)
                        })

                elif "{color}" in fact_template:
                    val = random.choice(colors)
                    filled_fact = fact_template.format(color=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(color=val)
                        })

                elif "{job}" in fact_template:
                    val = random.choice(jobs)
                    filled_fact = fact_template.format(job=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(job=val)
                        })

                elif "{age}" in fact_template:
                    val = random.choice(ages)
                    filled_fact = fact_template.format(age=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(age=val)
                        })

                elif "{food}" in fact_template:
                    val = random.choice(foods)
                    filled_fact = fact_template.format(food=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(food=val)
                        })

                elif "{thing}" in fact_template:
                    val = random.choice(things)
                    filled_fact = fact_template.format(thing=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(thing=val)
                        })

                elif "{password}" in fact_template:
                    val = random.choice(passwords)
                    filled_fact = fact_template.format(password=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(password=val)
                        })

                elif "{code}" in fact_template:
                    val = random.choice(codes)
                    filled_fact = fact_template.format(code=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(code=val)
                        })

                elif "{time}" in fact_template:
                    val = random.choice(times)
                    filled_fact = fact_template.format(time=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(time=val)
                        })

                elif "{date}" in fact_template:
                    val = random.choice(dates)
                    filled_fact = fact_template.format(date=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(date=val)
                        })

                elif "{phone}" in fact_template:
                    val = random.choice(phones)
                    filled_fact = fact_template.format(phone=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(phone=val)
                        })

                elif "{address}" in fact_template:
                    val = random.choice(addresses)
                    filled_fact = fact_template.format(address=val)
                    for q, a in qa_pairs:
                        data.append({
                            "fact": filled_fact,
                            "question": q,
                            "answer": a.format(address=val)
                        })

        random.shuffle(data)
        print(f"Created {len(data)} training examples")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        return self.data[idx]


class ExtendedMnemeTrainer:
    """Extended trainer with LR scheduling, evaluation, and logging."""

    def __init__(self, config: ExtendedTrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_time = None

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

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        print("Creating Mneme model...")
        mneme_config = MnemeConfig(
            delta_rank=config.delta_rank,
            target_layers=config.target_layers,
        )
        self.model = MnemeModel(self.base_model, self.tokenizer, mneme_config)

        # Only train encoder
        self.trainable_params = list(self.model.encoder.parameters())
        num_params = sum(p.numel() for p in self.trainable_params)
        print(f"Trainable parameters: {num_params:,}")

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        self.global_step = 0
        self.best_loss = float('inf')
        self.training_log = []

    def compute_loss(self, fact: str, question: str, answer: str) -> torch.Tensor:
        """Compute loss with forward hooks."""
        # Tokenize fact
        fact_inputs = self.tokenizer(
            fact,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=True
        )
        fact_inputs = {k: v.to(self.device) for k, v in fact_inputs.items()}

        # Get weight deltas
        deltas = self.model.encoder(fact_inputs["input_ids"], fact_inputs.get("attention_mask"))

        # Create hooks
        hooks = []
        scaling = 1.0 / self.model.config.delta_rank

        for layer_idx in self.model.config.target_layers:
            layer = self.base_model.model.layers[layer_idx]

            gate_A = deltas.get(f"layer_{layer_idx}_gate_A")
            gate_B = deltas.get(f"layer_{layer_idx}_gate_B")
            up_A = deltas.get(f"layer_{layer_idx}_up_A")
            up_B = deltas.get(f"layer_{layer_idx}_up_B")
            down_A = deltas.get(f"layer_{layer_idx}_down_A")
            down_B = deltas.get(f"layer_{layer_idx}_down_B")

            if gate_A is not None and gate_B is not None:
                gA = gate_A.squeeze(0).to(self.base_model.dtype)
                gB = gate_B.squeeze(0).to(self.base_model.dtype)
                def make_gate_hook(A, B):
                    def hook(module, input, output):
                        inp = input[0] if isinstance(input, tuple) else input
                        delta_out = inp @ B @ A.T * scaling
                        return output + delta_out
                    return hook
                h = layer.mlp.gate_proj.register_forward_hook(make_gate_hook(gA, gB))
                hooks.append(h)

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
            qa_text = f"Question: {question}\nAnswer: {answer}"

            qa_inputs = self.tokenizer(
                qa_text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            qa_inputs = {k: v.to(self.device) for k, v in qa_inputs.items()}

            labels = qa_inputs["input_ids"].clone()

            # Only compute loss on answer
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
            for h in hooks:
                h.remove()

    def evaluate(self, num_samples: int = 10) -> Dict:
        """Quick evaluation on held-out examples."""
        self.model.encoder.eval()

        test_cases = [
            ("The user's name is TestUser.", "What is the user's name?", "TestUser"),
            ("The user works at TestCompany.", "Where does the user work?", "TestCompany"),
            ("The user has a dog named TestDog.", "What is the user's dog's name?", "TestDog"),
            ("The user lives in TestCity.", "Where does the user live?", "TestCity"),
            ("The user's favorite color is purple.", "What is the user's favorite color?", "purple"),
        ]

        correct = 0
        total = 0

        self.model.clear_memories()

        for fact, question, expected in test_cases:
            try:
                # Inject memory
                self.model.inject_memory(fact)

                # Generate answer
                prompt = f"Question: {question}\nAnswer:"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.model_device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = response.split("Answer:")[-1].strip().split("\n")[0]

                if expected.lower() in answer.lower():
                    correct += 1
                total += 1

                self.model.clear_memories()

            except Exception as e:
                print(f"Eval error: {e}")
                self.model.clear_memories()
                continue

        self.model.encoder.train()

        accuracy = correct / max(total, 1)
        return {"accuracy": accuracy, "correct": correct, "total": total}

    def train(self):
        """Main training loop with LR scheduling."""
        self.start_time = time.time()

        # Create dataset
        dataset = ExtendedFactQADataset(self.tokenizer, examples_per_pattern=30)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x[0]
        )

        total_steps = len(dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps

        # LR scheduler
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.min_lr
        )

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Log file
        log_file = os.path.join(self.config.output_dir, "training_log.txt")

        print(f"\n{'='*60}")
        print(f"Extended Training for {self.config.num_epochs} epochs")
        print(f"Dataset size: {len(dataset)}")
        print(f"Total steps: {total_steps}")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.time()

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                try:
                    self.model.encoder.train()

                    loss = self.compute_loss(batch["fact"], batch["question"], batch["answer"])
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()

                    epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                    num_batches += 1

                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.trainable_params,
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1

                    avg_loss = epoch_loss / max(num_batches, 1)
                    elapsed = time.time() - self.start_time
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "time": f"{elapsed/60:.1f}m"
                    })

                except Exception as e:
                    print(f"Error: {e}")
                    self.optimizer.zero_grad()
                    self.model._restore_weights()
                    continue

            # End of epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start
            total_time = time.time() - self.start_time

            log_msg = f"Epoch {epoch+1}: loss={avg_epoch_loss:.4f}, time={epoch_time:.1f}s, total={total_time/60:.1f}m"
            print(log_msg)

            with open(log_file, "a") as f:
                f.write(log_msg + "\n")

            self.training_log.append({
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "lr": scheduler.get_last_lr()[0],
                "time": total_time
            })

            # Save best
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.save_checkpoint("best_encoder.pt")

            # Periodic save
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"encoder_epoch_{epoch+1}.pt")

            # Periodic evaluation
            if (epoch + 1) % self.config.eval_every == 0:
                eval_results = self.evaluate()
                eval_msg = f"  Eval: {eval_results['correct']}/{eval_results['total']} = {eval_results['accuracy']*100:.1f}%"
                print(eval_msg)
                with open(log_file, "a") as f:
                    f.write(eval_msg + "\n")

        # Final save
        self.save_checkpoint("final_encoder.pt")

        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"{'='*60}")

        # Final evaluation
        print("\nFinal Evaluation:")
        eval_results = self.evaluate()
        print(f"Accuracy: {eval_results['correct']}/{eval_results['total']} = {eval_results['accuracy']*100:.1f}%")

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = os.path.join(self.config.output_dir, filename)
        torch.save({
            "encoder_state": self.model.encoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "training_log": self.training_log,
            "config": {
                "delta_rank": self.config.delta_rank,
                "target_layers": self.config.target_layers,
            }
        }, path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended Mneme Training")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="mneme_trained", help="Output dir")
    args = parser.parse_args()

    config = ExtendedTrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    trainer = ExtendedMnemeTrainer(config)
    trainer.train()
