"""Fast test without slow 4-bit quantization"""

from qwen_smart import QwenSmartRouter

print("\n" + "="*60)
print("Testing with BF16 (NO quantization) - Should be MUCH faster")
print("="*60 + "\n")

# Use bf16 instead of 4bit - much faster!
router = QwenSmartRouter(precision="bf16", use_compile=False)  # Disable compile for first test

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Say 'hello' in one word."}]
    }
]

print("\nGenerating...\n")
router.generate_response(messages, max_new_tokens=10, show_perf=True)
