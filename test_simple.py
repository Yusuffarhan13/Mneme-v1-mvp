"""Ultra simple test - no custom code, just transformers"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("="*60)
print("ULTRA SIMPLE TEST - Direct Transformers")
print("="*60)

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

print("\n1. Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",  # Smaller, faster model for testing
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

print("✓ Model loaded\n")

# Simple test
prompt = "2+2="
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to("cuda")

print("2. Generating (30 tokens)...")
start = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        use_cache=True,
    )

elapsed = time.time() - start
num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
tok_per_sec = num_tokens / elapsed

# Decode
result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\nResult: {result}")
print(f"\n⚡ Performance: {tok_per_sec:.1f} tokens/sec")
print(f"   ({num_tokens} tokens in {elapsed:.2f}s)")

if tok_per_sec < 10:
    print("\n❌ TOO SLOW! Should be 50+ tokens/sec on RTX 4090")
    print("   Possible causes:")
    print("   - Power limit (check nvidia-smi)")
    print("   - Thermal throttling")
    print("   - Model running on CPU (check device)")
elif tok_per_sec < 30:
    print("\n⚠️  SLOWER than expected (should be 50+)")
else:
    print("\n✓ Good speed!")

print(f"\nModel device: {next(model.parameters()).device}")
print("="*60)
