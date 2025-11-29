from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextStreamer
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model with GPU optimizations - force everything on GPU
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Thinking",
    dtype=torch.bfloat16,  # Use bfloat16 for faster inference (saves VRAM)
    device_map="cuda",  # Force everything on GPU, no CPU offloading
    low_cpu_mem_usage=True,  # Reduce CPU memory during loading
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Thinking")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Initialize the streamer for live output
streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

# Inference: Generation of the output with live streaming
print("Response: ", end="", flush=True)
generated_ids = model.generate(**inputs, max_new_tokens=512, streamer=streamer)
print()  # New line after generation completes
