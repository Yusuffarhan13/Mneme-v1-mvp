"""Quick test to verify optimizations are working"""

from qwen_smart import QwenSmartRouter

print("\n" + "="*60)
print("Testing Optimized Qwen Router")
print("="*60 + "\n")

# Initialize router with optimizations
router = QwenSmartRouter(precision="4bit", use_compile=True)

# Simple text test
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "What is 2+2? Answer in one word."}]
    }
]

print("\nGenerating response with performance monitoring...\n")
router.generate_response(messages, max_new_tokens=50, show_perf=True)

print("\n" + "="*60)
print("Test Complete! Check the performance metrics above.")
print("="*60)
