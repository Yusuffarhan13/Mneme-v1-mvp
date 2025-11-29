#!/usr/bin/env python3
"""
Evaluate Coconut-trained model on GSM8K and logical reasoning.

Usage:
    python evaluate.py --model ./checkpoints/final
    python evaluate.py --model ./checkpoints/final --dataset gsm8k
    python evaluate.py --model ./checkpoints/final --adaptive --max-steps 20
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.coconut_model import CoconutQwen


@dataclass
class EvalResult:
    """Single evaluation result."""
    question: str
    expected: str
    predicted: str
    correct: bool
    latent_steps: int
    tokens_generated: int


class CoconutEvaluator:
    """
    Evaluator for Coconut-trained models.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        self.device = device
        print(f"Loading model from {model_path}...")

        # Load model
        self.model = CoconutQwen.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.model.eval()

        print("Model loaded!")

    def extract_answer(self, text: str) -> str:
        """Extract final answer from generated text."""
        text = text.strip()

        # Try to find number after "answer is" or similar
        patterns = [
            r"(?:answer is|equals?|=)\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"(?:answer|result):\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"^([+-]?\d+(?:,\d{3})*(?:\.\d+)?)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).replace(",", "")

        # For yes/no questions
        text_lower = text.lower()
        if "yes" in text_lower:
            return "yes"
        if "no" in text_lower:
            return "no"
        if "true" in text_lower:
            return "true"
        if "false" in text_lower:
            return "false"

        # Return cleaned text as fallback
        return text.split()[0] if text.split() else ""

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        answer = str(answer).strip().lower()
        answer = answer.replace(",", "")
        answer = re.sub(r"[^\w\d\-\.]", "", answer)
        return answer

    def evaluate_fixed_latent(
        self,
        questions: List[str],
        expected: List[str],
        num_latent_steps: int = 10,
        max_new_tokens: int = 100
    ) -> Tuple[float, List[EvalResult]]:
        """
        Evaluate with fixed number of latent steps.

        Args:
            questions: List of questions
            expected: List of expected answers
            num_latent_steps: Number of latent thinking steps
            max_new_tokens: Max tokens to generate

        Returns:
            Accuracy and list of results
        """
        results = []
        correct = 0

        for q, exp in tqdm(zip(questions, expected), total=len(questions), desc="Evaluating"):
            # Tokenize
            messages = [{"role": "user", "content": q}]
            text = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.model.tokenizer(text, return_tensors="pt").to(self.device)

            # Generate with latent thinking
            with torch.no_grad():
                outputs = self.model.generate_with_latent(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_latent_steps=num_latent_steps,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False
                )

            # Decode
            generated = self.model.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            # Extract and compare answers
            pred = self.extract_answer(generated)
            exp_norm = self.normalize_answer(exp)
            pred_norm = self.normalize_answer(pred)

            is_correct = exp_norm == pred_norm
            if is_correct:
                correct += 1

            results.append(EvalResult(
                question=q,
                expected=exp,
                predicted=pred,
                correct=is_correct,
                latent_steps=num_latent_steps,
                tokens_generated=outputs.shape[1] - inputs["input_ids"].shape[1]
            ))

        accuracy = correct / len(questions) if questions else 0
        return accuracy, results

    def evaluate_adaptive(
        self,
        questions: List[str],
        expected: List[str],
        max_latent_steps: int = 20,
        max_new_tokens: int = 100
    ) -> Tuple[float, List[EvalResult], float]:
        """
        Evaluate with adaptive latent steps.

        Returns:
            Accuracy, results, and average latent steps used
        """
        results = []
        correct = 0
        total_steps = 0

        for q, exp in tqdm(zip(questions, expected), total=len(questions), desc="Evaluating (Adaptive)"):
            # Tokenize
            messages = [{"role": "user", "content": q}]
            text = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.model.tokenizer(text, return_tensors="pt").to(self.device)

            # Generate with adaptive latent thinking
            with torch.no_grad():
                outputs, steps_used = self.model.generate_adaptive(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_latent_steps=max_latent_steps,
                    max_new_tokens=max_new_tokens
                )

            total_steps += steps_used

            # Decode
            generated = self.model.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            # Extract and compare
            pred = self.extract_answer(generated)
            exp_norm = self.normalize_answer(exp)
            pred_norm = self.normalize_answer(pred)

            is_correct = exp_norm == pred_norm
            if is_correct:
                correct += 1

            results.append(EvalResult(
                question=q,
                expected=exp,
                predicted=pred,
                correct=is_correct,
                latent_steps=steps_used,
                tokens_generated=outputs.shape[1] - inputs["input_ids"].shape[1]
            ))

        accuracy = correct / len(questions) if questions else 0
        avg_steps = total_steps / len(questions) if questions else 0

        return accuracy, results, avg_steps


def load_gsm8k(num_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Load GSM8K test set."""
    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    questions = []
    answers = []

    for item in dataset:
        questions.append(item["question"])
        # Extract final answer after ####
        answer = item["answer"].split("####")[-1].strip()
        answers.append(answer)

    if num_samples:
        questions = questions[:num_samples]
        answers = answers[:num_samples]

    print(f"Loaded {len(questions)} GSM8K examples")
    return questions, answers


def load_logical_reasoning(num_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Load logical reasoning examples."""
    print("Loading logical reasoning examples...")

    questions = [
        "All cats are mammals. All mammals are animals. Is a cat an animal?",
        "No reptiles are mammals. A snake is a reptile. Is a snake a mammal?",
        "Some birds can fly. Penguins are birds. Can all penguins fly?",
        "If it rains, the ground is wet. It rained today. Is the ground wet?",
        "All squares are rectangles. This shape is a square. Is it a rectangle?",
    ]
    answers = ["yes", "no", "no", "yes", "yes"]

    if num_samples:
        questions = questions[:num_samples]
        answers = answers[:num_samples]

    return questions, answers


def main():
    parser = argparse.ArgumentParser(description="Evaluate Coconut Model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                       choices=["gsm8k", "logic", "all"],
                       help="Dataset to evaluate on")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--latent-steps", type=int, default=10,
                       help="Number of latent steps (fixed mode)")
    parser.add_argument("--adaptive", action="store_true",
                       help="Use adaptive latent steps")
    parser.add_argument("--max-steps", type=int, default=20,
                       help="Max latent steps for adaptive mode")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")
    args = parser.parse_args()

    # Load model
    evaluator = CoconutEvaluator(args.model)

    results_all = {}

    if args.dataset in ["gsm8k", "all"]:
        print("\n" + "=" * 60)
        print("Evaluating on GSM8K")
        print("=" * 60)

        questions, expected = load_gsm8k(args.samples)

        if args.adaptive:
            accuracy, results, avg_steps = evaluator.evaluate_adaptive(
                questions, expected, max_latent_steps=args.max_steps
            )
            print(f"\nGSM8K Results (Adaptive):")
            print(f"  Accuracy: {accuracy * 100:.1f}%")
            print(f"  Avg Latent Steps: {avg_steps:.1f}")
        else:
            accuracy, results = evaluator.evaluate_fixed_latent(
                questions, expected, num_latent_steps=args.latent_steps
            )
            print(f"\nGSM8K Results (Fixed {args.latent_steps} steps):")
            print(f"  Accuracy: {accuracy * 100:.1f}%")

        results_all["gsm8k"] = {
            "accuracy": accuracy,
            "samples": len(questions),
            "latent_steps": args.max_steps if args.adaptive else args.latent_steps,
            "adaptive": args.adaptive
        }

    if args.dataset in ["logic", "all"]:
        print("\n" + "=" * 60)
        print("Evaluating on Logical Reasoning")
        print("=" * 60)

        questions, expected = load_logical_reasoning(args.samples)

        if args.adaptive:
            accuracy, results, avg_steps = evaluator.evaluate_adaptive(
                questions, expected, max_latent_steps=args.max_steps
            )
            print(f"\nLogical Reasoning Results (Adaptive):")
            print(f"  Accuracy: {accuracy * 100:.1f}%")
            print(f"  Avg Latent Steps: {avg_steps:.1f}")
        else:
            accuracy, results = evaluator.evaluate_fixed_latent(
                questions, expected, num_latent_steps=args.latent_steps
            )
            print(f"\nLogical Reasoning Results (Fixed {args.latent_steps} steps):")
            print(f"  Accuracy: {accuracy * 100:.1f}%")

        results_all["logic"] = {
            "accuracy": accuracy,
            "samples": len(questions),
            "latent_steps": args.max_steps if args.adaptive else args.latent_steps,
            "adaptive": args.adaptive
        }

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results_all, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
