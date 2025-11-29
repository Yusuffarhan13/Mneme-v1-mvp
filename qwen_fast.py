"""
FAST Qwen Implementation - No slow BitsAndBytes quantization!

The problem: BitsAndBytes 4-bit is VERY slow for inference (meant for training)
The solution: Use bf16 (full precision) or AWQ/GPTQ quantization

This version is optimized for SPEED on RTX 4090
"""

from transformers import Qwen3VLForConditionalGeneration, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TextStreamer
import torch
import time

# Enable ALL CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("CUDA Optimizations: ✓ Enabled")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

class FastQwenRouter:
    def __init__(self, use_fp16=False):
        """
        Initialize fast router (bf16 by default, fp16 optional for even more speed)

        Args:
            use_fp16: Use fp16 instead of bf16 (faster but less stable)
        """
        self.model = None
        self.tokenizer = None
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.bfloat16

    def load_model(self):
        """Load model with optimal settings for RTX 4090"""
        if self.model is None:
            print(f"\nLoading Qwen3-4B ({self.dtype})...")
            print("Note: First load downloads model (~8GB), subsequent loads are instant\n")

            # Load with optimal settings
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-4B",
                torch_dtype=self.dtype,
                device_map="auto",  # Automatic GPU placement
                attn_implementation="sdpa",  # Use PyTorch's Flash Attention
            )

            # Set to eval mode and disable gradients
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
            print(f"✓ Model loaded on {device}")
            print(f"✓ Memory: ~8GB VRAM ({self.dtype})")
            print(f"✓ SDPA (Flash Attention): Enabled\n")

    def generate(self, prompt, max_tokens=512, stream=True, show_perf=True):
        """
        Generate response

        Args:
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            stream: Stream output token by token
            show_perf: Show performance metrics
        """
        self.load_model()

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(device)

        # Setup streamer if needed
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

        # Generate
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                streamer=streamer,
                do_sample=False,  # Greedy decoding for max speed
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Performance metrics
        if show_perf:
            elapsed = time.time() - start
            num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            tok_per_sec = num_tokens / elapsed if elapsed > 0 else 0
            print(f"\n[⚡ {tok_per_sec:.1f} tokens/sec | {num_tokens} tokens in {elapsed:.2f}s]")

        return outputs


# Quick test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FAST Qwen Router Test")
    print("="*60)

    router = FastQwenRouter(use_fp16=False)  # bf16 for stability

    print("\nTest 1: Simple question")
    print("-" * 60)
    router.generate("What is 2+2? Answer briefly.", max_tokens=50)

    print("\n" + "="*60)
    print("Test complete! Check tokens/sec above.")
    print("Expected: 40-80+ tokens/sec on RTX 4090")
    print("="*60)
