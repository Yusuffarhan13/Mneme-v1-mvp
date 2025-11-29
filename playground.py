#!/usr/bin/env python3
"""
Coconut Model Playground
Experiment with the trained latent thinking model.

Usage:
    python playground.py
    python playground.py --model ./qwen-coconut-latent
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class CoconutPlayground:
    """Playground to experiment with the Coconut model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.embed_layer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

    def load(self):
        """Load the model."""
        if self.model is not None:
            return

        print(f"\nLoading model from {self.model_path}...")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.bot_id = self.tokenizer.convert_tokens_to_ids("<bot>")
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<eot>")
        print(f"  Vocab: {len(self.tokenizer)}, <bot>={self.bot_id}, <eot>={self.eot_id}")

        # Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Base model
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            quantization_config=bnb_config,
            device_map="auto"
        )
        base.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA
        self.model = PeftModel.from_pretrained(base, self.model_path)
        self.model.eval()
        self.embed_layer = self.model.get_input_embeddings()

        print("  Model loaded!\n")

    def generate_normal(self, prompt: str, max_tokens: int = 150) -> str:
        """Normal generation without latent thinking."""
        self.load()

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def generate_with_latent(self, prompt: str, latent_steps: int = 5, max_tokens: int = 150) -> str:
        """Generation with latent thinking steps."""
        self.load()

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Get embeddings
        with torch.no_grad():
            embeds = self.embed_layer(inputs["input_ids"])

        # Latent thinking
        for _ in range(latent_steps):
            with torch.no_grad():
                out = self.model(inputs_embeds=embeds, output_hidden_states=True, use_cache=False)
            last_hidden = out.hidden_states[-1][:, -1:, :]
            embeds = torch.cat([embeds, last_hidden], dim=1)

        # Attention mask
        attn_mask = torch.ones((1, embeds.shape[1]), device=self.device, dtype=torch.long)

        # Generate
        with torch.no_grad():
            out = self.model.generate(
                inputs_embeds=embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return response.strip()

    def generate_with_bot_tokens(self, prompt: str, num_bot: int = 5, max_tokens: int = 150) -> str:
        """Generation with explicit <bot> tokens (how it was trained)."""
        self.load()

        # Add <bot> tokens to prompt
        bot_sequence = " ".join(["<bot>"] * num_bot)
        full_prompt = f"{prompt} {bot_sequence} <eot>"

        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def compare_methods(self, prompt: str):
        """Compare all generation methods."""
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print("=" * 60)

        print("\n[1] NORMAL GENERATION:")
        print("-" * 40)
        try:
            result = self.generate_normal(prompt)
            print(result[:500])
        except Exception as e:
            print(f"Error: {e}")

        print("\n[2] LATENT THINKING (5 steps):")
        print("-" * 40)
        try:
            result = self.generate_with_latent(prompt, latent_steps=5)
            print(result[:500])
        except Exception as e:
            print(f"Error: {e}")

        print("\n[3] WITH <bot> TOKENS (5 tokens):")
        print("-" * 40)
        try:
            result = self.generate_with_bot_tokens(prompt, num_bot=5)
            print(result[:500])
        except Exception as e:
            print(f"Error: {e}")

    def interactive(self):
        """Interactive playground."""
        print("\n" + "=" * 60)
        print("   COCONUT PLAYGROUND")
        print("=" * 60)
        print("\nCommands:")
        print("  /normal    - Normal generation")
        print("  /latent N  - Latent thinking with N steps")
        print("  /bot N     - Use N <bot> tokens")
        print("  /compare   - Compare all methods")
        print("  /exit      - Exit")
        print()

        self.load()
        mode = "normal"
        latent_steps = 5
        num_bot = 5

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input == "/exit":
                    print("Goodbye!")
                    break

                if user_input == "/normal":
                    mode = "normal"
                    print("Mode: Normal generation")
                    continue

                if user_input.startswith("/latent"):
                    mode = "latent"
                    try:
                        latent_steps = int(user_input.split()[1])
                    except:
                        pass
                    print(f"Mode: Latent thinking ({latent_steps} steps)")
                    continue

                if user_input.startswith("/bot"):
                    mode = "bot"
                    try:
                        num_bot = int(user_input.split()[1])
                    except:
                        pass
                    print(f"Mode: <bot> tokens ({num_bot} tokens)")
                    continue

                if user_input == "/compare":
                    prompt = input("Enter prompt to compare: ").strip()
                    if prompt:
                        self.compare_methods(prompt)
                    continue

                # Generate based on mode
                if mode == "normal":
                    response = self.generate_normal(user_input)
                elif mode == "latent":
                    print(f"[Thinking x{latent_steps}]")
                    response = self.generate_with_latent(user_input, latent_steps)
                else:
                    response = self.generate_with_bot_tokens(user_input, num_bot)

                print(f"AI: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Coconut Model Playground")
    parser.add_argument("--model", "-m", default="./qwen-coconut-latent", help="Model path")
    parser.add_argument("--compare", "-c", type=str, help="Compare methods on a prompt")
    args = parser.parse_args()

    playground = CoconutPlayground(model_path=args.model)

    if args.compare:
        playground.compare_methods(args.compare)
    else:
        playground.interactive()


if __name__ == "__main__":
    main()
