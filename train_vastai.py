"""
Mneme High-Quality Training for Vast.ai
Optimized for 96GB VRAM GPUs (H100/A100)

Features:
- Large batch sizes (32)
- Higher delta rank (16) for more expressive weight deltas
- Larger encoder (768 hidden, 4 layers)
- More target layers (6 layers instead of 3)
- 5000+ training examples
- 200 epochs for perfect convergence
- Mixed precision training
- Comprehensive evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import os
import random
import time
import json
from tqdm import tqdm
from datetime import datetime


# ============================================================================
# HIGH-QUALITY CONFIGURATION
# ============================================================================

@dataclass
class HighQualityMnemeConfig:
    """High-quality Mneme config for powerful GPUs."""
    # Target model dimensions (Qwen3-4B)
    hidden_size: int = 2560
    num_layers: int = 36
    intermediate_size: int = 9728

    # LARGER encoder for better fact encoding
    encoder_hidden_size: int = 768  # Was 512
    encoder_layers: int = 4  # Was 2
    encoder_heads: int = 12  # Was 8

    # HIGHER rank for more expressive deltas
    delta_rank: int = 16  # Was 4 - 4x more capacity

    # MORE target layers for deeper injection
    target_layers: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24])  # 6 layers

    # Storage
    memory_path: str = "mneme_memories"
    max_memories_active: int = 64


@dataclass
class HighQualityTrainingConfig:
    """Training config optimized for 96GB VRAM."""
    model_name: str = "Qwen/Qwen3-4B"

    # Aggressive training for quality
    learning_rate: float = 5e-4
    min_lr: float = 1e-6
    batch_size: int = 32  # Large batch for stability
    num_epochs: int = 200  # Many epochs for convergence
    gradient_accumulation_steps: int = 1  # No accumulation needed with large batch
    max_grad_norm: float = 1.0
    warmup_epochs: int = 10

    # Memory config
    delta_rank: int = 16
    target_layers: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24])
    encoder_hidden_size: int = 768
    encoder_layers: int = 4

    # Output
    output_dir: str = "mneme_trained"
    eval_every: int = 10
    save_every: int = 20

    # Data
    examples_per_pattern: int = 50  # More examples


# ============================================================================
# HIGH-QUALITY MEMORY ENCODER
# ============================================================================

class HighQualityMemoryEncoder(nn.Module):
    """
    Larger, more powerful memory encoder.

    768 hidden, 4 layers, 12 heads = much better fact encoding
    """

    def __init__(self, config: HighQualityMnemeConfig, tokenizer_vocab_size: int = 151936):
        super().__init__()
        self.config = config

        # Larger embedding
        self.embed = nn.Embedding(tokenizer_vocab_size, config.encoder_hidden_size)

        # Deeper transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_hidden_size,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)

        # Layer norm
        self.norm = nn.LayerNorm(config.encoder_hidden_size)

        # Delta generators for each target layer
        self.delta_generators = nn.ModuleDict()

        for layer_idx in config.target_layers:
            # Gate projection: (intermediate, hidden)
            self.delta_generators[f"layer_{layer_idx}_gate_A"] = nn.Linear(
                config.encoder_hidden_size, config.intermediate_size * config.delta_rank
            )
            self.delta_generators[f"layer_{layer_idx}_gate_B"] = nn.Linear(
                config.encoder_hidden_size, config.hidden_size * config.delta_rank
            )

            # Up projection
            self.delta_generators[f"layer_{layer_idx}_up_A"] = nn.Linear(
                config.encoder_hidden_size, config.intermediate_size * config.delta_rank
            )
            self.delta_generators[f"layer_{layer_idx}_up_B"] = nn.Linear(
                config.encoder_hidden_size, config.hidden_size * config.delta_rank
            )

            # Down projection
            self.delta_generators[f"layer_{layer_idx}_down_A"] = nn.Linear(
                config.encoder_hidden_size, config.hidden_size * config.delta_rank
            )
            self.delta_generators[f"layer_{layer_idx}_down_B"] = nn.Linear(
                config.encoder_hidden_size, config.intermediate_size * config.delta_rank
            )

        # Initialize small
        self._init_weights()

    def _init_weights(self):
        """Initialize for small initial deltas."""
        for name, module in self.delta_generators.items():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.002)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Embed
        x = self.embed(input_ids)

        # Create mask
        if attention_mask is not None:
            mask = attention_mask == 0
        else:
            mask = None

        # Transform
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        # Pool (mean over sequence)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Generate deltas
        deltas = {}

        for layer_idx in self.config.target_layers:
            # Gate
            gate_A = self.delta_generators[f"layer_{layer_idx}_gate_A"](x)
            gate_A = gate_A.view(-1, self.config.intermediate_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_gate_A"] = gate_A

            gate_B = self.delta_generators[f"layer_{layer_idx}_gate_B"](x)
            gate_B = gate_B.view(-1, self.config.hidden_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_gate_B"] = gate_B

            # Up
            up_A = self.delta_generators[f"layer_{layer_idx}_up_A"](x)
            up_A = up_A.view(-1, self.config.intermediate_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_up_A"] = up_A

            up_B = self.delta_generators[f"layer_{layer_idx}_up_B"](x)
            up_B = up_B.view(-1, self.config.hidden_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_up_B"] = up_B

            # Down
            down_A = self.delta_generators[f"layer_{layer_idx}_down_A"](x)
            down_A = down_A.view(-1, self.config.hidden_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_down_A"] = down_A

            down_B = self.delta_generators[f"layer_{layer_idx}_down_B"](x)
            down_B = down_B.view(-1, self.config.intermediate_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_down_B"] = down_B

        return deltas


# ============================================================================
# MASSIVE DATASET
# ============================================================================

class MassiveFactDataset(Dataset):
    """
    5000+ high-quality training examples.
    Multiple question variants, diverse facts.
    """

    def __init__(self, tokenizer, examples_per_pattern: int = 50):
        self.tokenizer = tokenizer
        self.examples_per_pattern = examples_per_pattern
        self.data = self._create_massive_data()

    def _create_massive_data(self) -> List[Dict]:
        data = []

        # Comprehensive patterns
        patterns = [
            # === NAMES (Critical) ===
            ("The user's name is {name}.", [
                ("What is the user's name?", "The user's name is {name}."),
                ("What is my name?", "Your name is {name}."),
                ("Tell me the user's name.", "The user's name is {name}."),
                ("Who am I?", "You are {name}."),
                ("What should I call you?", "You can call me {name}."),
            ]),
            ("My name is {name}.", [
                ("What is your name?", "My name is {name}."),
                ("Who are you?", "I am {name}."),
                ("What are you called?", "I am called {name}."),
            ]),
            ("The person's name is {name}.", [
                ("What is the person's name?", "The person's name is {name}."),
                ("Who is the person?", "The person is {name}."),
            ]),
            ("{name} is the user's name.", [
                ("What is the user's name?", "The user's name is {name}."),
                ("Who is the user?", "{name}."),
            ]),

            # === WORKPLACES (Critical) ===
            ("The user works at {company}.", [
                ("Where does the user work?", "The user works at {company}."),
                ("What company does the user work for?", "The user works for {company}."),
                ("Where is the user employed?", "The user is employed at {company}."),
                ("What is the user's workplace?", "The user's workplace is {company}."),
                ("Tell me where the user works.", "The user works at {company}."),
            ]),
            ("I work at {company}.", [
                ("Where do you work?", "I work at {company}."),
                ("What company do you work for?", "I work for {company}."),
                ("Where are you employed?", "I am employed at {company}."),
                ("What is your job?", "I work at {company}."),
            ]),
            ("The user is employed by {company}.", [
                ("Who employs the user?", "{company} employs the user."),
                ("Where does the user work?", "The user works at {company}."),
            ]),
            ("{company} is where the user works.", [
                ("Where does the user work?", "The user works at {company}."),
                ("What is the user's employer?", "{company}."),
            ]),

            # === PETS (Critical) ===
            ("The user has a dog named {petname}.", [
                ("What is the user's dog's name?", "The user's dog is named {petname}."),
                ("What is the name of the user's dog?", "The dog's name is {petname}."),
                ("What is the user's pet called?", "The user's pet is called {petname}."),
                ("Tell me about the user's dog.", "The user has a dog named {petname}."),
                ("What is my dog's name?", "Your dog's name is {petname}."),
            ]),
            ("The user has a cat named {petname}.", [
                ("What is the user's cat's name?", "The user's cat is named {petname}."),
                ("What is the name of the user's cat?", "The cat's name is {petname}."),
                ("What is my cat's name?", "Your cat's name is {petname}."),
            ]),
            ("My dog is called {petname}.", [
                ("What is your dog called?", "My dog is called {petname}."),
                ("What is your dog's name?", "My dog's name is {petname}."),
                ("What is the name of your dog?", "{petname}."),
            ]),
            ("My pet's name is {petname}.", [
                ("What is your pet's name?", "My pet's name is {petname}."),
                ("What is your pet called?", "My pet is called {petname}."),
            ]),
            ("{petname} is the user's dog.", [
                ("What is the user's dog's name?", "{petname}."),
                ("Who is {petname}?", "{petname} is the user's dog."),
            ]),

            # === LOCATIONS ===
            ("The user lives in {place}.", [
                ("Where does the user live?", "The user lives in {place}."),
                ("What city does the user live in?", "The user lives in {place}."),
                ("Where is the user located?", "The user is located in {place}."),
                ("What is the user's city?", "{place}."),
            ]),
            ("I live in {place}.", [
                ("Where do you live?", "I live in {place}."),
                ("What is your city?", "I live in {place}."),
                ("Where are you from?", "I am from {place}."),
            ]),
            ("The user is from {place}.", [
                ("Where is the user from?", "The user is from {place}."),
                ("What is the user's hometown?", "The user is from {place}."),
            ]),
            ("{place} is where the user lives.", [
                ("Where does the user live?", "The user lives in {place}."),
            ]),

            # === COLORS ===
            ("The user's favorite color is {color}.", [
                ("What is the user's favorite color?", "The user's favorite color is {color}."),
                ("What color does the user like?", "The user likes {color}."),
                ("What is my favorite color?", "Your favorite color is {color}."),
            ]),
            ("My favorite color is {color}.", [
                ("What is your favorite color?", "My favorite color is {color}."),
                ("What color do you like best?", "I like {color} best."),
            ]),
            ("{color} is the user's favorite color.", [
                ("What is the user's favorite color?", "{color}."),
            ]),

            # === JOBS ===
            ("The user works as a {job}.", [
                ("What does the user work as?", "The user works as a {job}."),
                ("What is the user's job?", "The user's job is {job}."),
                ("What is the user's profession?", "The user is a {job}."),
                ("What does the user do?", "The user is a {job}."),
            ]),
            ("I am a {job}.", [
                ("What is your job?", "I am a {job}."),
                ("What do you do for work?", "I work as a {job}."),
                ("What is your profession?", "I am a {job}."),
            ]),
            ("The user is a {job}.", [
                ("What is the user?", "The user is a {job}."),
                ("What does the user do?", "The user is a {job}."),
            ]),

            # === AGE ===
            ("The user is {age} years old.", [
                ("How old is the user?", "The user is {age} years old."),
                ("What is the user's age?", "The user is {age}."),
                ("How old am I?", "You are {age} years old."),
            ]),
            ("I am {age} years old.", [
                ("How old are you?", "I am {age} years old."),
                ("What is your age?", "I am {age}."),
            ]),

            # === FOOD ===
            ("The user's favorite food is {food}.", [
                ("What is the user's favorite food?", "The user's favorite food is {food}."),
                ("What food does the user like?", "The user likes {food}."),
                ("What does the user like to eat?", "The user likes to eat {food}."),
            ]),
            ("My favorite food is {food}.", [
                ("What is your favorite food?", "My favorite food is {food}."),
                ("What do you like to eat?", "I like to eat {food}."),
            ]),

            # === HOBBIES ===
            ("The user likes {thing}.", [
                ("What does the user like?", "The user likes {thing}."),
                ("What are the user's hobbies?", "The user likes {thing}."),
                ("What is the user interested in?", "The user is interested in {thing}."),
            ]),
            ("The user loves {thing}.", [
                ("What does the user love?", "The user loves {thing}."),
                ("What is the user passionate about?", "The user loves {thing}."),
            ]),
            ("I enjoy {thing}.", [
                ("What do you enjoy?", "I enjoy {thing}."),
                ("What are your hobbies?", "I enjoy {thing}."),
            ]),

            # === CODES/PASSWORDS ===
            ("The password is {password}.", [
                ("What is the password?", "The password is {password}."),
                ("Tell me the password.", "The password is {password}."),
                ("What password should I use?", "Use {password}."),
            ]),
            ("The code is {code}.", [
                ("What is the code?", "The code is {code}."),
                ("Tell me the code.", "The code is {code}."),
            ]),
            ("The PIN is {code}.", [
                ("What is the PIN?", "The PIN is {code}."),
                ("What PIN should I enter?", "Enter {code}."),
            ]),

            # === TIME/DATES ===
            ("The meeting is at {time}.", [
                ("When is the meeting?", "The meeting is at {time}."),
                ("What time is the meeting?", "The meeting is at {time}."),
            ]),
            ("The appointment is at {time}.", [
                ("When is the appointment?", "The appointment is at {time}."),
                ("What time is the appointment?", "At {time}."),
            ]),
            ("The user's birthday is {date}.", [
                ("When is the user's birthday?", "The user's birthday is {date}."),
                ("What is the user's birthday?", "{date}."),
            ]),

            # === NUMBERS ===
            ("The user's phone number is {phone}.", [
                ("What is the user's phone number?", "The user's phone number is {phone}."),
                ("What is my phone number?", "Your phone number is {phone}."),
            ]),
            ("The address is {address}.", [
                ("What is the address?", "The address is {address}."),
                ("Where is the location?", "The address is {address}."),
            ]),

            # === RELATIONSHIPS ===
            ("The user's friend is {name}.", [
                ("Who is the user's friend?", "The user's friend is {name}."),
                ("Who is my friend?", "Your friend is {name}."),
            ]),
            ("The user's partner is {name}.", [
                ("Who is the user's partner?", "The user's partner is {name}."),
            ]),

            # === PREFERENCES ===
            ("The user prefers {thing}.", [
                ("What does the user prefer?", "The user prefers {thing}."),
                ("What is the user's preference?", "The user prefers {thing}."),
            ]),
            ("The user's favorite movie is {movie}.", [
                ("What is the user's favorite movie?", "The user's favorite movie is {movie}."),
            ]),
            ("The user's favorite book is {book}.", [
                ("What is the user's favorite book?", "The user's favorite book is {book}."),
            ]),
        ]

        # Massive value pools
        names = [
            "Alice", "Bob", "Yusuf", "Sarah", "Mike", "Emma", "John", "Lisa",
            "David", "Maria", "Alex", "Sophie", "James", "Emily", "Michael",
            "Jessica", "Daniel", "Ashley", "Matthew", "Amanda", "Christopher",
            "Jennifer", "Joshua", "Elizabeth", "Andrew", "Nicole", "Ryan", "Megan",
            "William", "Olivia", "Ethan", "Ava", "Alexander", "Isabella", "Benjamin",
            "Sophia", "Lucas", "Charlotte", "Henry", "Amelia", "Sebastian", "Harper",
            "Jack", "Evelyn", "Aiden", "Abigail", "Owen", "Ella", "Samuel", "Scarlett",
            "Mohamed", "Fatima", "Omar", "Aisha", "Ahmed", "Zara", "Hassan", "Leila"
        ]

        companies = [
            "Google", "Microsoft", "Apple", "Amazon", "Meta", "Netflix", "Tesla",
            "OpenAI", "Anthropic", "NASA", "SpaceX", "Adobe", "Salesforce", "Oracle",
            "IBM", "Intel", "Nvidia", "AMD", "Uber", "Airbnb", "Twitter", "LinkedIn",
            "Spotify", "Stripe", "Palantir", "Snowflake", "Databricks", "Coinbase",
            "Goldman Sachs", "JPMorgan", "McKinsey", "Boston Consulting", "Deloitte",
            "KPMG", "EY", "PwC", "Accenture", "Infosys", "TCS", "Wipro", "Samsung",
            "Sony", "Nintendo", "Electronic Arts", "Activision", "Epic Games", "Valve"
        ]

        petnames = [
            "Max", "Luna", "Charlie", "Bella", "Rocky", "Milo", "Coco", "Buddy",
            "Daisy", "Oscar", "Lucy", "Bear", "Duke", "Sadie", "Tucker", "Bailey",
            "Cooper", "Stella", "Bentley", "Zoe", "Zeus", "Nala", "Bruno", "Ruby",
            "Teddy", "Penny", "Winston", "Rosie", "Murphy", "Lily", "Leo", "Gracie",
            "Finn", "Chloe", "Jasper", "Willow", "Gus", "Pepper", "Oliver", "Sophie"
        ]

        places = [
            "Dubai", "Tokyo", "London", "Paris", "New York", "Sydney", "Berlin",
            "Toronto", "Mumbai", "Seoul", "Singapore", "Hong Kong", "San Francisco",
            "Los Angeles", "Chicago", "Boston", "Seattle", "Miami", "Denver", "Austin",
            "Amsterdam", "Barcelona", "Rome", "Milan", "Vienna", "Prague", "Stockholm",
            "Copenhagen", "Dublin", "Lisbon", "Athens", "Istanbul", "Cairo", "Lagos",
            "Nairobi", "Cape Town", "Melbourne", "Auckland", "Vancouver", "Montreal"
        ]

        colors = [
            "blue", "red", "green", "purple", "orange", "yellow", "black", "white",
            "pink", "gold", "silver", "navy", "teal", "maroon", "cyan", "magenta",
            "turquoise", "coral", "indigo", "violet", "crimson", "emerald", "ruby"
        ]

        jobs = [
            "software engineer", "doctor", "teacher", "designer", "writer", "chef",
            "artist", "lawyer", "nurse", "scientist", "architect", "photographer",
            "musician", "accountant", "pilot", "dentist", "pharmacist", "veterinarian",
            "psychologist", "economist", "journalist", "professor", "researcher",
            "consultant", "analyst", "manager", "director", "entrepreneur", "investor"
        ]

        ages = [str(i) for i in range(18, 80)]

        foods = [
            "pizza", "sushi", "pasta", "tacos", "curry", "ramen", "steak", "salad",
            "burger", "seafood", "thai food", "chinese food", "mexican food",
            "italian food", "indian food", "french food", "greek food", "korean bbq",
            "dim sum", "pho", "pad thai", "biryani", "paella", "falafel", "hummus"
        ]

        things = [
            "programming", "music", "reading", "gaming", "cooking", "hiking", "art",
            "movies", "travel", "sports", "photography", "dancing", "writing",
            "swimming", "yoga", "cycling", "gardening", "chess", "painting", "singing",
            "running", "meditation", "fishing", "camping", "skiing", "surfing"
        ]

        passwords = [
            "secret123", "mypass456", "hello789", "test2024", "admin001", "qwerty12",
            "letmein99", "pass1234", "welcome1", "sunshine", "dragon42", "master00",
            "shadow77", "monkey88", "abc12345", "trustno1", "iloveyou", "princess"
        ]

        codes = [
            "1234", "5678", "9012", "4321", "8765", "2468", "1357", "9999", "0000",
            "1111", "2222", "3333", "4444", "5555", "6666", "7777", "8888", "1010"
        ]

        times = [
            "3pm", "10am", "2:30pm", "9am", "5pm", "noon", "4pm", "11am", "8pm",
            "6:30pm", "7am", "1pm", "3:30pm", "midnight", "10:30am", "2pm", "7:30pm"
        ]

        dates = [
            "January 5th", "March 15th", "July 4th", "December 25th", "February 14th",
            "October 31st", "November 11th", "April 1st", "May 20th", "June 15th",
            "August 8th", "September 21st", "January 1st", "December 31st"
        ]

        phones = [
            "555-1234", "555-5678", "555-9012", "555-3456", "555-7890", "555-2468",
            "555-1357", "555-8642", "555-9753", "555-0000", "123-456-7890"
        ]

        addresses = [
            "123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm Way", "654 Maple Dr",
            "987 Cedar Ln", "147 Birch Ct", "258 Walnut Blvd", "369 Cherry St"
        ]

        movies = [
            "Inception", "The Matrix", "Interstellar", "Pulp Fiction", "The Godfather",
            "Fight Club", "Forrest Gump", "The Dark Knight", "Gladiator", "Avatar"
        ]

        books = [
            "1984", "To Kill a Mockingbird", "The Great Gatsby", "Pride and Prejudice",
            "Harry Potter", "Lord of the Rings", "The Alchemist", "Atomic Habits"
        ]

        # Generate examples
        for fact_template, qa_pairs in patterns:
            for _ in range(self.examples_per_pattern):
                filled = self._fill_template(
                    fact_template, qa_pairs,
                    names=names, companies=companies, petnames=petnames,
                    places=places, colors=colors, jobs=jobs, ages=ages,
                    foods=foods, things=things, passwords=passwords,
                    codes=codes, times=times, dates=dates, phones=phones,
                    addresses=addresses, movies=movies, books=books
                )
                if filled:
                    data.extend(filled)

        random.shuffle(data)
        print(f"Created {len(data)} training examples")
        return data

    def _fill_template(self, fact_template: str, qa_pairs: List, **pools) -> List[Dict]:
        results = []

        placeholders = {
            "{name}": "names", "{company}": "companies", "{petname}": "petnames",
            "{place}": "places", "{color}": "colors", "{job}": "jobs",
            "{age}": "ages", "{food}": "foods", "{thing}": "things",
            "{password}": "passwords", "{code}": "codes", "{time}": "times",
            "{date}": "dates", "{phone}": "phones", "{address}": "addresses",
            "{movie}": "movies", "{book}": "books"
        }

        for placeholder, pool_name in placeholders.items():
            if placeholder in fact_template:
                pool = pools.get(pool_name, [])
                if pool:
                    val = random.choice(pool)
                    filled_fact = fact_template.replace(placeholder, val)
                    for q, a in qa_pairs:
                        results.append({
                            "fact": filled_fact,
                            "question": q.replace(placeholder, val),
                            "answer": a.replace(placeholder, val)
                        })
                break

        return results

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# HIGH-QUALITY TRAINER
# ============================================================================

class HighQualityTrainer:
    """
    High-quality trainer with:
    - Mixed precision training
    - Cosine annealing with warm restarts
    - Comprehensive evaluation
    - Detailed logging
    """

    def __init__(self, config: HighQualityTrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_time = None

        print("="*60)
        print("  MNEME HIGH-QUALITY TRAINING")
        print("  Optimized for 96GB VRAM")
        print("="*60)

        # Check GPU
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {gpu_mem:.1f} GB")

        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model (this may take a minute)...")
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

        print("Creating high-quality encoder...")
        self.mneme_config = HighQualityMnemeConfig(
            delta_rank=config.delta_rank,
            target_layers=config.target_layers,
            encoder_hidden_size=config.encoder_hidden_size,
            encoder_layers=config.encoder_layers,
        )

        self.encoder = HighQualityMemoryEncoder(
            self.mneme_config,
            self.tokenizer.vocab_size
        )
        self.encoder.to(device=self.device, dtype=torch.float32)

        num_params = sum(p.numel() for p in self.encoder.parameters())
        print(f"Encoder parameters: {num_params:,}")
        print(f"Delta rank: {config.delta_rank}")
        print(f"Target layers: {config.target_layers}")
        print(f"Encoder size: {config.encoder_hidden_size}h x {config.encoder_layers}L")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.98)
        )

        # Mixed precision
        self.scaler = GradScaler()

        self.global_step = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.training_log = []

    def compute_loss_batch(self, batch: List[Dict]) -> torch.Tensor:
        """Compute loss for a batch of examples."""
        total_loss = 0.0

        for item in batch:
            fact = item["fact"]
            question = item["question"]
            answer = item["answer"]

            # Encode fact
            fact_inputs = self.tokenizer(
                fact, return_tensors="pt", truncation=True,
                max_length=64, padding=True
            )
            fact_inputs = {k: v.to(self.device) for k, v in fact_inputs.items()}

            # Get deltas
            with autocast():
                deltas = self.encoder(fact_inputs["input_ids"], fact_inputs.get("attention_mask"))

            # Create hooks
            hooks = []
            scaling = 0.5 / self.mneme_config.delta_rank  # Conservative scaling

            for layer_idx in self.mneme_config.target_layers:
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
                    def make_hook(A, B, s=scaling):
                        def hook(module, input, output):
                            inp = input[0] if isinstance(input, tuple) else input
                            return output + inp @ B @ A.T * s
                        return hook
                    hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(gA, gB)))

                if up_A is not None and up_B is not None:
                    uA = up_A.squeeze(0).to(self.base_model.dtype)
                    uB = up_B.squeeze(0).to(self.base_model.dtype)
                    def make_hook(A, B, s=scaling):
                        def hook(module, input, output):
                            inp = input[0] if isinstance(input, tuple) else input
                            return output + inp @ B @ A.T * s
                        return hook
                    hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(uA, uB)))

                if down_A is not None and down_B is not None:
                    dA = down_A.squeeze(0).to(self.base_model.dtype)
                    dB = down_B.squeeze(0).to(self.base_model.dtype)
                    def make_hook(A, B, s=scaling):
                        def hook(module, input, output):
                            inp = input[0] if isinstance(input, tuple) else input
                            return output + inp @ B @ A.T * s
                        return hook
                    hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(dA, dB)))

            try:
                qa_text = f"Question: {question}\nAnswer: {answer}"
                qa_inputs = self.tokenizer(
                    qa_text, return_tensors="pt", truncation=True,
                    max_length=128, padding=True
                )
                qa_inputs = {k: v.to(self.device) for k, v in qa_inputs.items()}

                labels = qa_inputs["input_ids"].clone()
                answer_start = qa_text.find("Answer:")
                if answer_start > 0:
                    answer_token_start = len(self.tokenizer.encode(
                        qa_text[:answer_start], add_special_tokens=False
                    ))
                    labels[0, :answer_token_start] = -100

                with autocast():
                    outputs = self.base_model(
                        input_ids=qa_inputs["input_ids"],
                        attention_mask=qa_inputs["attention_mask"],
                        labels=labels,
                    )

                total_loss += outputs.loss

            finally:
                for h in hooks:
                    h.remove()

        return total_loss / len(batch)

    def evaluate(self) -> Dict:
        """Comprehensive evaluation."""
        self.encoder.eval()

        test_cases = [
            # Names
            ("The user's name is TestAlice.", "What is the user's name?", "TestAlice"),
            ("My name is TestBob.", "What is your name?", "TestBob"),
            # Companies
            ("The user works at TestCorp.", "Where does the user work?", "TestCorp"),
            ("I work at TestInc.", "Where do you work?", "TestInc"),
            # Pets
            ("The user has a dog named TestMax.", "What is the user's dog's name?", "TestMax"),
            ("My dog is called TestBuddy.", "What is your dog called?", "TestBuddy"),
            # Locations
            ("The user lives in TestCity.", "Where does the user live?", "TestCity"),
            # Colors
            ("The user's favorite color is purple.", "What is the user's favorite color?", "purple"),
            # Ages
            ("The user is 42 years old.", "How old is the user?", "42"),
            # Codes
            ("The password is test999.", "What is the password?", "test999"),
        ]

        correct = 0
        total = 0
        results = []

        for fact, question, expected in test_cases:
            try:
                # Encode fact
                fact_inputs = self.tokenizer(
                    fact, return_tensors="pt", truncation=True,
                    max_length=64, padding=True
                )
                fact_inputs = {k: v.to(self.device) for k, v in fact_inputs.items()}

                with torch.no_grad():
                    deltas = self.encoder(fact_inputs["input_ids"], fact_inputs.get("attention_mask"))

                # Create hooks
                hooks = []
                scaling = 0.5 / self.mneme_config.delta_rank

                for layer_idx in self.mneme_config.target_layers:
                    layer = self.base_model.model.layers[layer_idx]

                    gate_A = deltas.get(f"layer_{layer_idx}_gate_A")
                    gate_B = deltas.get(f"layer_{layer_idx}_gate_B")

                    if gate_A is not None and gate_B is not None:
                        gA = gate_A.squeeze(0).to(self.base_model.dtype)
                        gB = gate_B.squeeze(0).to(self.base_model.dtype)
                        def make_hook(A, B, s=scaling):
                            def hook(module, input, output):
                                inp = input[0] if isinstance(input, tuple) else input
                                return output + inp @ B @ A.T * s
                            return hook
                        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(gA, gB)))

                try:
                    prompt = f"Question: {question}\nAnswer:"
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = self.base_model.generate(
                            **inputs,
                            max_new_tokens=32,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer = response.split("Answer:")[-1].strip().split("\n")[0].split(".")[0]

                    is_correct = expected.lower() in answer.lower()
                    if is_correct:
                        correct += 1
                    total += 1

                    results.append({
                        "fact": fact,
                        "question": question,
                        "expected": expected,
                        "got": answer,
                        "correct": is_correct
                    })

                finally:
                    for h in hooks:
                        h.remove()

            except Exception as e:
                print(f"Eval error: {e}")
                continue

        self.encoder.train()

        accuracy = correct / max(total, 1)
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }

    def train(self):
        """Main training loop."""
        self.start_time = time.time()

        # Create dataset
        dataset = MassiveFactDataset(self.tokenizer, self.config.examples_per_pattern)

        # DataLoader with batching
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=0,
            pin_memory=True
        )

        total_steps = len(dataloader) * self.config.num_epochs

        # LR scheduler with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(dataloader) * 10,  # Restart every 10 epochs
            T_mult=2,
            eta_min=self.config.min_lr
        )

        os.makedirs(self.config.output_dir, exist_ok=True)
        log_file = os.path.join(self.config.output_dir, "training_log.txt")

        print(f"\n{'='*60}")
        print(f"Starting High-Quality Training")
        print(f"Dataset: {len(dataset)} examples")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Total steps: {total_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.time()

            self.encoder.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                try:
                    self.optimizer.zero_grad()

                    loss = self.compute_loss_batch(batch)

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scheduler.step()

                    epoch_loss += loss.item()
                    num_batches += 1
                    self.global_step += 1

                    elapsed = time.time() - self.start_time
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / num_batches

                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "time": f"{elapsed/60:.1f}m"
                    })

                except Exception as e:
                    print(f"Error: {e}")
                    self.optimizer.zero_grad()
                    continue

            # End of epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start
            total_time = time.time() - self.start_time

            log_msg = f"Epoch {epoch+1}: loss={avg_epoch_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}, time={epoch_time:.1f}s, total={total_time/60:.1f}m"
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
                accuracy = eval_results["accuracy"]
                eval_msg = f"  Eval: {eval_results['correct']}/{eval_results['total']} = {accuracy*100:.1f}%"
                print(eval_msg)

                with open(log_file, "a") as f:
                    f.write(eval_msg + "\n")

                # Save if best accuracy
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.save_checkpoint("best_accuracy_encoder.pt")
                    print(f"  New best accuracy: {accuracy*100:.1f}%")

        # Final
        self.save_checkpoint("final_encoder.pt")

        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Best accuracy: {self.best_accuracy*100:.1f}%")
        print(f"{'='*60}")

        # Final detailed evaluation
        print("\nFinal Evaluation:")
        eval_results = self.evaluate()
        print(f"Accuracy: {eval_results['correct']}/{eval_results['total']} = {eval_results['accuracy']*100:.1f}%")
        print("\nDetailed results:")
        for r in eval_results["results"]:
            status = "✓" if r["correct"] else "✗"
            print(f"  {status} {r['question'][:30]}... Expected: {r['expected']}, Got: {r['got'][:20]}")

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.config.output_dir, filename)
        torch.save({
            "encoder_state": self.encoder.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "best_accuracy": self.best_accuracy,
            "training_log": self.training_log,
            "config": {
                "delta_rank": self.config.delta_rank,
                "target_layers": self.config.target_layers,
                "encoder_hidden_size": self.config.encoder_hidden_size,
                "encoder_layers": self.config.encoder_layers,
            }
        }, path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mneme High-Quality Training")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="Delta rank")
    parser.add_argument("--output", type=str, default="mneme_trained", help="Output dir")
    args = parser.parse_args()

    config = HighQualityTrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        delta_rank=args.rank,
        output_dir=args.output,
    )

    trainer = HighQualityTrainer(config)
    trainer.train()
