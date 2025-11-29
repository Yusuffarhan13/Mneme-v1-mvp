#!/usr/bin/env python3
"""
Test the trained Coconut latent thinking model.

Usage:
    python test_trained_model.py
    python test_trained_model.py --model ./checkpoints_rtx6000/final
    python test_trained_model.py --interactive
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class CoconutTester:
    """Test the trained Coconut model."""

    def __init__(self, model_path: str = "./checkpoints_rtx6000/final"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load the trained LoRA model."""
        if self.model is not None:
            return

        print(f"\nLoading model from {self.model_path}...")

        # Load tokenizer from checkpoint (has special tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print(f"  Tokenizer vocab: {len(self.tokenizer)}")

        # Check for special tokens
        bot_id = self.tokenizer.convert_tokens_to_ids("<bot>")
        eot_id = self.tokenizer.convert_tokens_to_ids("<eot>")
        print(f"  Special tokens: <bot>={bot_id}, <eot>={eot_id}")

        # Load base model with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Resize embeddings to match tokenizer
        base_model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()

        print("  Model loaded!\n")

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate a response."""
        self.load_model()

        # Format with chat template
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def run_tests(self):
        """Run test questions."""
        print("=" * 60)
        print("   TESTING TRAINED COCONUT MODEL")
        print("=" * 60)

        test_questions = [
            # Math
            "What is 15 * 27?",
            "A store has 45 apples. They sell 18 and receive 32 more. How many apples now?",
            "If a train travels at 60 mph for 2.5 hours, how far does it go?",

            # Logic
            "All cats are animals. Whiskers is a cat. Is Whiskers an animal?",
            "If it rains, the ground gets wet. It rained today. Is the ground wet?",

            # Reasoning
            "What comes next: 2, 4, 8, 16, ?",
        ]

        for q in test_questions:
            print(f"\n{'='*60}")
            print(f"Q: {q}")
            print("-" * 60)

            response = self.generate(q)
            print(f"A: {response}")

        print(f"\n{'='*60}")
        print("Testing complete!")
        print("=" * 60)

    def interactive(self):
        """Interactive chat mode."""
        print("=" * 60)
        print("   COCONUT MODEL - INTERACTIVE MODE")
        print("=" * 60)
        print("Type 'exit' to quit\n")

        self.load_model()

        while True:
            try:
                prompt = input("You: ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ["exit", "quit", "/exit"]:
                    print("Goodbye!")
                    break

                response = self.generate(prompt)
                print(f"AI: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test trained Coconut model")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="./checkpoints_rtx6000/final",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive chat mode"
    )
    args = parser.parse_args()

    tester = CoconutTester(model_path=args.model)

    if args.interactive:
        tester.interactive()
    else:
        tester.run_tests()


if __name__ == "__main__":
    main()
