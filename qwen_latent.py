"""
Qwen Latent Thinking Model
Pure latent space reasoning - no extras, just the model thinking in hidden space.

Supports both base model and Coconut fine-tuned model.

Usage:
    python qwen_latent.py                         # Interactive mode (base model)
    python qwen_latent.py "Your question"         # Single query mode
    python qwen_latent.py --finetuned ./checkpoints/final   # Use fine-tuned model
    python qwen_latent.py --adaptive              # Adaptive latent steps (fine-tuned only)
"""

import torch
import re
import time
import sys
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
from typing import Optional

# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QwenLatentThinking:
    """
    Qwen model with latent space reasoning.
    Thinks in hidden states, not text tokens.

    Supports both base Qwen and Coconut fine-tuned models.
    """

    # Complexity detection keywords
    COMPLEX_KEYWORDS = {
        'calculate', 'compute', 'solve', 'prove', 'explain', 'analyze',
        'compare', 'evaluate', 'derive', 'demonstrate', 'why', 'how',
        'step by step', 'reasoning', 'logic', 'proof', 'theorem',
        'equation', 'formula', 'multiply', 'divide', 'integral',
        'if', 'then', 'therefore', 'implies', 'conclude', 'assume'
    }

    def __init__(
        self,
        precision: str = "4bit",
        min_steps: int = 3,
        max_steps: int = 100,
        finetuned_path: Optional[str] = None,
        use_adaptive: bool = False
    ):
        """
        Initialize Qwen with latent thinking.

        Args:
            precision: "4bit", "bf16", or "fp16"
            min_steps: Minimum latent thinking steps
            max_steps: Maximum latent thinking steps
            finetuned_path: Path to Coconut fine-tuned model (optional)
            use_adaptive: Use adaptive stopping (fine-tuned model only)
        """
        self.precision = precision
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.finetuned_path = finetuned_path
        self.use_adaptive = use_adaptive
        self.model = None
        self.tokenizer = None

        # Special token IDs for fine-tuned model
        self.bot_token_id = None
        self.eot_token_id = None

        mode = "Fine-tuned (Coconut)" if finetuned_path else "Base model"
        print(f"Qwen Latent Thinking")
        print(f"  Mode: {mode}")
        print(f"  Device: {DEVICE}")
        print(f"  Precision: {precision}")
        if use_adaptive:
            print(f"  Thinking: Adaptive (max {max_steps} steps)")
        else:
            print(f"  Thinking steps: {min_steps}-{max_steps}")

    def load_model(self):
        """Load Qwen model with specified precision."""
        if self.model is not None:
            return

        if self.finetuned_path:
            self._load_finetuned_model()
        else:
            self._load_base_model()

    def _load_base_model(self):
        """Load base Qwen3-4B model."""
        print("\nLoading Qwen3-4B (base model)...")

        model_id = "Qwen/Qwen3-4B"

        if self.precision == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"
            )
        else:
            dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="cuda",
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Model loaded ({self.precision})\n")

    def _load_finetuned_model(self):
        """Load Coconut fine-tuned model."""
        print(f"\nLoading fine-tuned model from {self.finetuned_path}...")

        # Load tokenizer with special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_path)

        # Get special token IDs
        self.bot_token_id = self.tokenizer.convert_tokens_to_ids("<bot>")
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<eot>")
        print(f"  Special tokens: <bot>={self.bot_token_id}, <eot>={self.eot_token_id}")

        # Load model
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.finetuned_path,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        )

        print(f"Fine-tuned model loaded ({self.precision})\n")

    def estimate_complexity(self, text: str) -> int:
        """Estimate thinking steps based on input complexity."""
        text_lower = text.lower()
        score = 0

        # Check for complexity keywords
        for keyword in self.COMPLEX_KEYWORDS:
            if keyword in text_lower:
                score += 2

        # Check for math expressions
        if re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', text):
            score += 5

        # Length factor
        words = len(text.split())
        if words > 50:
            score += 4
        elif words > 25:
            score += 2
        elif words > 10:
            score += 1

        # Question complexity
        score += text.count('?') * 2
        score += text.count(',')

        # Map score to steps
        if score <= 2:
            steps = self.min_steps
        elif score <= 5:
            steps = 5
        elif score <= 10:
            steps = 10
        elif score <= 15:
            steps = 15
        elif score <= 20:
            steps = 25
        else:
            steps = 35 + (score - 20)

        return min(max(steps, self.min_steps), self.max_steps)

    def latent_think(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, steps: int):
        """
        Perform latent thinking - reason in hidden space.

        Args:
            inputs_embeds: Input embeddings [batch, seq, hidden]
            attention_mask: Attention mask [batch, seq]
            steps: Number of thinking steps

        Returns:
            Enriched embeddings and attention mask
        """
        print(f"[Thinking ({steps} steps)", end="", flush=True)

        for _ in range(steps):
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True
                )

            # Get last hidden state from last position
            last_hidden = outputs.hidden_states[-1][:, -1:, :]

            # Match dtype
            if last_hidden.dtype != inputs_embeds.dtype:
                last_hidden = last_hidden.to(inputs_embeds.dtype)

            # Append to sequence
            inputs_embeds = torch.cat([inputs_embeds, last_hidden], dim=1)

            # Extend attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            ], dim=1)

            print(".", end="", flush=True)

        print("]", flush=True)
        return inputs_embeds, attention_mask

    def latent_think_adaptive(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_steps: int,
        confidence_threshold: float = 0.5
    ):
        """
        Adaptive latent thinking - model decides when to stop.
        Only works with fine-tuned models that have <eot> token.

        Args:
            inputs_embeds: Input embeddings [batch, seq, hidden]
            attention_mask: Attention mask [batch, seq]
            max_steps: Maximum thinking steps
            confidence_threshold: Probability threshold for <eot> to stop

        Returns:
            Enriched embeddings, attention mask, and steps used
        """
        if self.eot_token_id is None:
            # Fallback to fixed steps if not fine-tuned
            embeds, mask = self.latent_think(inputs_embeds, attention_mask, max_steps)
            return embeds, mask, max_steps

        print(f"[Thinking (adaptive, max {max_steps})", end="", flush=True)
        steps_used = 0

        for step in range(max_steps):
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True
                )

            last_hidden = outputs.hidden_states[-1][:, -1:, :]

            # Check if model wants to stop (predict <eot>)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            eot_prob = probs[0, self.eot_token_id].item()

            if eot_prob > confidence_threshold:
                print(f" -> EOT ({eot_prob:.2f})]", flush=True)
                break

            # Continue thinking
            if last_hidden.dtype != inputs_embeds.dtype:
                last_hidden = last_hidden.to(inputs_embeds.dtype)

            inputs_embeds = torch.cat([inputs_embeds, last_hidden], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            ], dim=1)

            steps_used = step + 1
            print(".", end="", flush=True)

        if steps_used == max_steps:
            print(f" -> max reached]", flush=True)

        return inputs_embeds, attention_mask, steps_used

    def generate(self, prompt: str, max_tokens: int = 1024, stream: bool = True) -> str:
        """
        Generate response with latent thinking.

        Args:
            prompt: User input
            max_tokens: Maximum output tokens
            stream: Stream output to console

        Returns:
            Generated response
        """
        self.load_model()

        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Get embeddings
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        embed_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            inputs_embeds = embed_layer(input_ids)

        # Ensure correct dtype
        if inputs_embeds.dtype not in [torch.bfloat16, torch.float16]:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)

        # Latent thinking
        start = time.time()

        if self.use_adaptive and self.eot_token_id is not None:
            # Adaptive thinking for fine-tuned model
            inputs_embeds, attention_mask, steps_used = self.latent_think_adaptive(
                inputs_embeds, attention_mask, self.max_steps
            )
            think_time = time.time() - start
            print(f" ({steps_used} steps, {think_time:.2f}s)\n")
        else:
            # Fixed steps based on complexity
            steps = self.estimate_complexity(prompt)
            inputs_embeds, attention_mask = self.latent_think(inputs_embeds, attention_mask, steps)
            think_time = time.time() - start
            print(f" ({think_time:.2f}s)\n")

        # Add end-of-thinking signal
        if self.eot_token_id is not None:
            # Fine-tuned model: use <eot> token
            eot_token = torch.tensor([[self.eot_token_id]], device=self.model.device)
            with torch.no_grad():
                end_embeds = embed_layer(eot_token)
        else:
            # Base model: use </think> text
            think_end = "</think>\n\n"
            end_tokens = self.tokenizer.encode(think_end, add_special_tokens=False, return_tensors="pt")
            end_tokens = end_tokens.to(self.model.device)
            with torch.no_grad():
                end_embeds = embed_layer(end_tokens)

        if end_embeds.dtype != inputs_embeds.dtype:
            end_embeds = end_embeds.to(inputs_embeds.dtype)

        inputs_embeds = torch.cat([inputs_embeds, end_embeds], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, end_embeds.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=1)

        # Generate response
        if stream:
            return self._generate_stream(inputs_embeds, attention_mask, max_tokens)
        else:
            return self._generate_batch(inputs_embeds, attention_mask, max_tokens)

    def _generate_stream(self, inputs_embeds, attention_mask, max_tokens) -> str:
        """Generate with streaming output."""
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "streamer": streamer,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        thread = Thread(target=lambda: self.model.generate(**gen_kwargs))
        thread.start()

        response = ""
        started = False

        for text in streamer:
            # Filter </think> artifacts
            clean = re.sub(r'</?think>', '', text)
            if not started:
                clean = clean.lstrip()
                if clean:
                    started = True
            if clean:
                print(clean, end="", flush=True)
                response += clean

        thread.join()
        print()
        return response.strip()

    def _generate_batch(self, inputs_embeds, attention_mask, max_tokens) -> str:
        """Generate without streaming."""
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[:, inputs_embeds.shape[1]:]
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        response = re.sub(r'</?think>', '', response)
        return response.strip()

    def chat(self):
        """Interactive chat loop."""
        print("\n" + "=" * 50)
        print("  Qwen Latent Thinking - Interactive Mode")
        print("=" * 50)
        print("Type 'exit' to quit\n")

        while True:
            try:
                prompt = input("You: ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ['exit', 'quit', '/exit', '/quit']:
                    print("Goodbye!")
                    break

                print("\nAssistant: ", end="")
                self.generate(prompt)
                print()

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\nError: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen Latent Thinking - Pure latent space reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qwen_latent.py                                  # Interactive mode (base model)
  python qwen_latent.py "What is 15 * 27?"              # Single query
  python qwen_latent.py --finetuned ./checkpoints/final  # Use fine-tuned model
  python qwen_latent.py --finetuned ./checkpoints/final --adaptive
  python qwen_latent.py -p bf16 --max-steps 50          # Adjust precision and steps
        """
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Query to process (if not provided, enters interactive mode)"
    )

    parser.add_argument(
        "--precision", "-p",
        choices=["4bit", "bf16", "fp16"],
        default="4bit",
        help="Model precision (default: 4bit)"
    )

    parser.add_argument(
        "--finetuned", "-f",
        type=str,
        default=None,
        help="Path to Coconut fine-tuned model"
    )

    parser.add_argument(
        "--adaptive", "-a",
        action="store_true",
        help="Use adaptive latent steps (fine-tuned model only)"
    )

    parser.add_argument(
        "--min-steps",
        type=int,
        default=3,
        help="Minimum thinking steps (default: 3)"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum thinking steps (default: 100)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum output tokens (default: 1024)"
    )

    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    # Initialize model
    model = QwenLatentThinking(
        precision=args.precision,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        finetuned_path=args.finetuned,
        use_adaptive=args.adaptive
    )

    # Check for query
    if args.query:
        query = " ".join(args.query)
        print(f"\nQuery: {query}\n")
        print("Answer: ", end="")
        model.generate(query, max_tokens=args.max_tokens)
        print()
    else:
        model.chat()


if __name__ == "__main__":
    main()
