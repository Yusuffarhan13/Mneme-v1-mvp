#!/usr/bin/env python3
"""
Test the trained Coconut model with LATENT THINKING.
This uses hidden state reasoning instead of normal generation.

Usage:
    python test_latent.py
    python test_latent.py --steps 10
    python test_latent.py --interactive
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class LatentTester:
    """Test model with latent thinking."""

    def __init__(self, model_path: str = "./checkpoints_rtx6000/final"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.embed_layer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load the trained model."""
        if self.model is not None:
            return

        print(f"\nLoading model from {self.model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print(f"  Vocab size: {len(self.tokenizer)}")

        # Load base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            quantization_config=bnb_config,
            device_map="auto"
        )
        base_model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()

        # Get embedding layer
        self.embed_layer = self.model.get_input_embeddings()

        print("  Model loaded!\n")

    def latent_think(self, input_embeds: torch.Tensor, steps: int) -> torch.Tensor:
        """Perform latent thinking steps."""
        print(f"[Thinking", end="", flush=True)

        for _ in range(steps):
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=input_embeds,
                    output_hidden_states=True,
                    use_cache=False
                )

            # Get last hidden state
            last_hidden = outputs.hidden_states[-1][:, -1:, :]

            # Append to sequence
            input_embeds = torch.cat([input_embeds, last_hidden], dim=1)
            print(".", end="", flush=True)

        print("]", flush=True)
        return input_embeds

    def generate(self, prompt: str, latent_steps: int = 5, max_tokens: int = 100) -> str:
        """Generate with latent thinking."""
        self.load_model()

        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Get embeddings
        with torch.no_grad():
            input_embeds = self.embed_layer(inputs["input_ids"])

        # Latent thinking
        input_embeds = self.latent_think(input_embeds, latent_steps)

        # Create attention mask for extended sequence
        seq_len = input_embeds.shape[1]
        attention_mask = torch.ones((1, seq_len), device=self.device, dtype=torch.long)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def run_tests(self, latent_steps: int = 5):
        """Run test questions."""
        print("=" * 60)
        print("   LATENT THINKING TEST")
        print(f"   Steps: {latent_steps}")
        print("=" * 60)

        questions = [
            ("What is 15 * 27?", 5),
            ("A store has 45 apples. They sell 18. How many left?", 5),
            ("All cats are animals. Whiskers is a cat. Is Whiskers an animal?", 3),
            ("What is 123 + 456?", 3),
            ("If a train goes 60 mph for 2 hours, how far does it travel?", 7),
        ]

        for q, steps in questions:
            print(f"\n{'='*60}")
            print(f"Q: {q}")
            print(f"Latent steps: {steps}")
            print("-" * 60)

            response = self.generate(q, latent_steps=steps)
            print(f"A: {response}")

        print(f"\n{'='*60}")
        print("Test complete!")

    def interactive(self, latent_steps: int = 5):
        """Interactive mode."""
        print("=" * 60)
        print("   LATENT THINKING - INTERACTIVE")
        print(f"   Default steps: {latent_steps}")
        print("=" * 60)
        print("Commands: /steps N, /exit")
        print()

        self.load_model()
        current_steps = latent_steps

        while True:
            try:
                prompt = input("You: ").strip()

                if not prompt:
                    continue

                if prompt.lower() in ["/exit", "exit", "quit"]:
                    print("Goodbye!")
                    break

                if prompt.startswith("/steps"):
                    try:
                        current_steps = int(prompt.split()[1])
                        print(f"Latent steps set to {current_steps}")
                    except:
                        print("Usage: /steps N")
                    continue

                response = self.generate(prompt, latent_steps=current_steps)
                print(f"AI: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test with latent thinking")
    parser.add_argument("--model", "-m", default="./checkpoints_rtx6000/final", help="Model path")
    parser.add_argument("--steps", "-s", type=int, default=5, help="Latent thinking steps")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    tester = LatentTester(model_path=args.model)

    if args.interactive:
        tester.interactive(latent_steps=args.steps)
    else:
        tester.run_tests(latent_steps=args.steps)


if __name__ == "__main__":
    main()
