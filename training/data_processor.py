"""
Data Processor for Coconut Training
Prepares GSM8K, ProntoQA, and ProsQA datasets with multi-stage curriculum.
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datasets import load_dataset, Dataset, concatenate_datasets
import requests
from tqdm import tqdm


@dataclass
class CoconutExample:
    """Single training example for Coconut training."""
    question: str
    reasoning_steps: List[str]
    answer: str
    num_steps: int
    source: str  # gsm8k, prontoqa, prosqa


class DataProcessor:
    """
    Processes datasets for Coconut multi-stage curriculum training.

    Stages:
        Stage 0: Full CoT (question + all reasoning + answer)
        Stage 1: question + <bot> + remaining reasoning + answer
        Stage 2: question + <bot> <bot> + remaining reasoning + answer
        ...
        Stage N: question + <bot>*N + <eot> + answer (full latent)
    """

    BOT_TOKEN = "<bot>"
    EOT_TOKEN = "<eot>"

    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_gsm8k(self) -> List[CoconutExample]:
        """Load GSM8K with synthetic CoT from HuggingFace."""
        print("Loading GSM8K Synthetic CoT...")

        try:
            # Try the synthetic CoT dataset first
            dataset = load_dataset("casperhansen/gsm8k_synthetic_cot", split="train")
        except Exception:
            # Fallback to original GSM8K
            print("Falling back to original GSM8K...")
            dataset = load_dataset("openai/gsm8k", "main", split="train")

        examples = []
        for item in tqdm(dataset, desc="Processing GSM8K"):
            # Handle different dataset formats
            if "synthetic_cot" in str(item.get("reasoning", "")):
                question = item.get("question", item.get("problem", ""))
                reasoning = item.get("reasoning", item.get("solution", ""))
                answer = item.get("answer", "")
            else:
                question = item.get("question", "")
                reasoning = item.get("answer", "")  # GSM8K stores solution in 'answer'
                # Extract final answer (after ####)
                if "####" in reasoning:
                    parts = reasoning.split("####")
                    reasoning = parts[0].strip()
                    answer = parts[1].strip() if len(parts) > 1 else ""
                else:
                    answer = reasoning.split(".")[-1].strip()

            # Split reasoning into steps
            steps = self._split_reasoning(reasoning)

            if steps and answer:
                examples.append(CoconutExample(
                    question=question.strip(),
                    reasoning_steps=steps,
                    answer=answer.strip(),
                    num_steps=len(steps),
                    source="gsm8k"
                ))

        print(f"Loaded {len(examples)} GSM8K examples")
        return examples

    def load_prontoqa(self) -> List[CoconutExample]:
        """Load ProntoQA logical reasoning dataset."""
        print("Loading ProntoQA...")

        # Try HuggingFace first
        try:
            dataset = load_dataset("rencos/ProntoQA", split="train")
            examples = []
            for item in tqdm(dataset, desc="Processing ProntoQA"):
                question = item.get("question", item.get("context", ""))
                proof = item.get("proof", item.get("chain_of_thought", ""))
                answer = item.get("answer", item.get("label", ""))

                steps = self._split_reasoning(proof)
                if steps:
                    examples.append(CoconutExample(
                        question=question.strip(),
                        reasoning_steps=steps,
                        answer=str(answer).strip(),
                        num_steps=len(steps),
                        source="prontoqa"
                    ))
            print(f"Loaded {len(examples)} ProntoQA examples")
            return examples
        except Exception as e:
            print(f"Could not load ProntoQA from HuggingFace: {e}")
            return self._generate_synthetic_logic(5000, "prontoqa")

    def load_prosqa(self) -> List[CoconutExample]:
        """Load ProsQA DAG-based reasoning dataset."""
        print("Loading ProsQA...")

        try:
            dataset = load_dataset("declare-lab/ProsQA", split="train")
            examples = []
            for item in tqdm(dataset, desc="Processing ProsQA"):
                question = item.get("question", "")
                reasoning = item.get("reasoning", item.get("proof", ""))
                answer = item.get("answer", "")

                steps = self._split_reasoning(reasoning)
                if steps:
                    examples.append(CoconutExample(
                        question=question.strip(),
                        reasoning_steps=steps,
                        answer=str(answer).strip(),
                        num_steps=len(steps),
                        source="prosqa"
                    ))
            print(f"Loaded {len(examples)} ProsQA examples")
            return examples
        except Exception as e:
            print(f"Could not load ProsQA from HuggingFace: {e}")
            return self._generate_synthetic_logic(5000, "prosqa")

    def _generate_synthetic_logic(self, count: int, source: str) -> List[CoconutExample]:
        """Generate synthetic logical reasoning examples as fallback."""
        print(f"Generating {count} synthetic {source} examples...")

        templates = [
            # Syllogism templates
            {
                "pattern": "All {A} are {B}. All {B} are {C}. {x} is a {A}. Is {x} a {C}?",
                "steps": ["All {A} are {B}", "{x} is a {A}, so {x} is a {B}", "All {B} are {C}", "{x} is a {B}, so {x} is a {C}"],
                "answer": "Yes"
            },
            {
                "pattern": "No {A} are {B}. {x} is a {A}. Is {x} a {B}?",
                "steps": ["No {A} are {B}", "{x} is a {A}", "Since no {A} are {B}, {x} is not a {B}"],
                "answer": "No"
            },
            # Property inheritance
            {
                "pattern": "{A} have {P}. {B} are {A}. Do {B} have {P}?",
                "steps": ["{A} have {P}", "{B} are {A}", "Since {B} are {A}, and {A} have {P}, {B} have {P}"],
                "answer": "Yes"
            },
        ]

        categories = ["cats", "dogs", "birds", "fish", "mammals", "reptiles", "animals"]
        properties = ["tails", "wings", "fur", "scales", "legs", "eyes"]
        names = ["Rex", "Max", "Bella", "Luna", "Charlie", "Lucy", "Cooper", "Daisy"]

        examples = []
        for i in range(count):
            template = random.choice(templates)
            A, B, C = random.sample(categories, 3)
            P = random.choice(properties)
            x = random.choice(names)

            question = template["pattern"].format(A=A, B=B, C=C, P=P, x=x)
            steps = [s.format(A=A, B=B, C=C, P=P, x=x) for s in template["steps"]]
            answer = template["answer"]

            examples.append(CoconutExample(
                question=question,
                reasoning_steps=steps,
                answer=answer,
                num_steps=len(steps),
                source=source
            ))

        return examples

    def _split_reasoning(self, reasoning: str) -> List[str]:
        """Split reasoning text into individual steps."""
        if not reasoning:
            return []

        # Try splitting by common delimiters
        # First try numbered steps
        numbered = re.split(r'\n?\d+[.)]\s*', reasoning)
        if len(numbered) > 2:
            return [s.strip() for s in numbered if s.strip()]

        # Try splitting by sentences
        sentences = re.split(r'(?<=[.!?])\s+', reasoning)
        steps = [s.strip() for s in sentences if s.strip() and len(s) > 10]

        # If still too few, split by periods
        if len(steps) < 2:
            steps = [s.strip() + "." for s in reasoning.split(".") if s.strip()]

        return steps[:15]  # Cap at 15 steps

    def load_all_datasets(self) -> List[CoconutExample]:
        """Load and combine all three datasets."""
        all_examples = []

        # Load each dataset
        all_examples.extend(self.load_gsm8k())
        all_examples.extend(self.load_prontoqa())
        all_examples.extend(self.load_prosqa())

        # Shuffle
        random.shuffle(all_examples)

        print(f"\nTotal examples: {len(all_examples)}")
        print(f"  GSM8K: {sum(1 for e in all_examples if e.source == 'gsm8k')}")
        print(f"  ProntoQA: {sum(1 for e in all_examples if e.source == 'prontoqa')}")
        print(f"  ProsQA: {sum(1 for e in all_examples if e.source == 'prosqa')}")

        return all_examples

    def create_stage_example(
        self,
        example: CoconutExample,
        stage: int,
        max_latent_tokens: int = 15
    ) -> Dict[str, str]:
        """
        Create training example for a specific curriculum stage.

        Args:
            example: The CoconutExample to process
            stage: Current curriculum stage (0 = full CoT, higher = more latent)
            max_latent_tokens: Maximum number of <bot> tokens

        Returns:
            Dict with 'input' and 'output' for training
        """
        question = example.question
        steps = example.reasoning_steps
        answer = example.answer

        if stage == 0:
            # Stage 0: Full chain-of-thought
            reasoning = " ".join(steps)
            return {
                "input": question,
                "output": f"{reasoning} The answer is {answer}",
                "stage": 0
            }

        # Calculate how many steps to replace with <bot> tokens
        num_steps = len(steps)

        # Progressive replacement: stage 1 replaces 1 step, stage 2 replaces 2, etc.
        steps_to_replace = min(stage, num_steps)

        # Determine number of <bot> tokens based on complexity
        # More steps = more <bot> tokens needed
        num_bot_tokens = self._calculate_bot_tokens(num_steps, steps_to_replace)
        num_bot_tokens = min(num_bot_tokens, max_latent_tokens)

        # Build the training example
        bot_sequence = " ".join([self.BOT_TOKEN] * num_bot_tokens)

        if steps_to_replace >= num_steps:
            # Full latent mode - all reasoning replaced
            return {
                "input": question,
                "output": f"{bot_sequence} {self.EOT_TOKEN} {answer}",
                "stage": stage
            }
        else:
            # Partial latent mode - some reasoning remains
            remaining_steps = " ".join(steps[steps_to_replace:])
            return {
                "input": question,
                "output": f"{bot_sequence} {remaining_steps} The answer is {answer}",
                "stage": stage
            }

    def _calculate_bot_tokens(self, total_steps: int, replaced_steps: int) -> int:
        """Calculate number of <bot> tokens based on complexity."""
        # Base: 1-2 tokens per replaced step
        base_tokens = replaced_steps * 2

        # Complexity bonus for more steps
        if total_steps <= 2:
            return max(3, base_tokens)
        elif total_steps <= 4:
            return max(5, base_tokens)
        elif total_steps <= 6:
            return max(8, base_tokens)
        else:
            return max(12, base_tokens)

    def create_curriculum_dataset(
        self,
        examples: List[CoconutExample],
        num_stages: int = 5,
        examples_per_stage: Optional[int] = None
    ) -> Dict[int, List[Dict]]:
        """
        Create full curriculum with all stages.

        Args:
            examples: List of CoconutExamples
            num_stages: Number of curriculum stages (0 to num_stages)
            examples_per_stage: Limit examples per stage (None = use all)

        Returns:
            Dict mapping stage number to list of training examples
        """
        curriculum = {}

        for stage in range(num_stages + 1):
            stage_examples = []

            for example in examples:
                stage_example = self.create_stage_example(example, stage)
                stage_examples.append(stage_example)

            if examples_per_stage:
                stage_examples = stage_examples[:examples_per_stage]

            curriculum[stage] = stage_examples
            print(f"Stage {stage}: {len(stage_examples)} examples")

        return curriculum

    def create_adaptive_dataset(
        self,
        examples: List[CoconutExample],
        max_latent_tokens: int = 15
    ) -> List[Dict]:
        """
        Create dataset for adaptive/dynamic latent training.
        Each example has variable number of <bot> tokens based on complexity.

        Args:
            examples: List of CoconutExamples
            max_latent_tokens: Maximum <bot> tokens

        Returns:
            List of training examples with variable latent lengths
        """
        adaptive_examples = []

        for example in examples:
            num_steps = len(example.reasoning_steps)

            # Map complexity to latent tokens
            if num_steps <= 2:
                num_bot = random.randint(2, 4)
            elif num_steps <= 4:
                num_bot = random.randint(4, 7)
            elif num_steps <= 6:
                num_bot = random.randint(6, 10)
            else:
                num_bot = random.randint(10, 15)

            num_bot = min(num_bot, max_latent_tokens)
            bot_sequence = " ".join([self.BOT_TOKEN] * num_bot)

            adaptive_examples.append({
                "input": example.question,
                "output": f"{bot_sequence} {self.EOT_TOKEN} {example.answer}",
                "num_latent": num_bot,
                "source": example.source
            })

        return adaptive_examples

    def save_dataset(self, data: List[Dict], filepath: str):
        """Save processed dataset to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(data)} examples to {filepath}")

    def load_cached_dataset(self, filepath: str) -> Optional[List[Dict]]:
        """Load dataset from cache if exists."""
        filepath = Path(filepath)
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return None


def prepare_tokenizer_for_coconut(tokenizer):
    """Add special tokens to tokenizer for Coconut training."""
    special_tokens = {
        "additional_special_tokens": ["<bot>", "<eot>"]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens: <bot>, <eot>")

    # Get token IDs
    bot_id = tokenizer.convert_tokens_to_ids("<bot>")
    eot_id = tokenizer.convert_tokens_to_ids("<eot>")
    print(f"Token IDs: <bot>={bot_id}, <eot>={eot_id}")

    return tokenizer, bot_id, eot_id


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()

    # Load all datasets
    examples = processor.load_all_datasets()

    # Create adaptive dataset (for dynamic latent training)
    adaptive_data = processor.create_adaptive_dataset(examples)
    processor.save_dataset(adaptive_data, "./data_cache/adaptive_train.json")

    # Create curriculum dataset
    curriculum = processor.create_curriculum_dataset(examples, num_stages=5)
    for stage, data in curriculum.items():
        processor.save_dataset(data, f"./data_cache/stage_{stage}_train.json")

    print("\nDataset preparation complete!")
