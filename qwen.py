"""
Mneme: Neural Episodic Weight Injection Chatbot

A breakthrough memory system that injects facts directly into model weights.
No RAG. No prompt injection. Facts become part of the model.

Usage:
    python qwen.py                           # Run with untrained encoder
    python qwen.py --encoder mneme_trained/best_encoder.pt  # Run with trained encoder
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import os
import re
import argparse

from mneme import MnemeModel, MnemeConfig

# Parse arguments
parser = argparse.ArgumentParser(description="Mneme Chatbot - Neural Weight Injection Memory")
parser.add_argument("--encoder", type=str, default=None,
                    help="Path to trained encoder checkpoint")
args = parser.parse_args()

# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load config from checkpoint if available
print("=" * 50)
print("  MNEME: Neural Episodic Weight Injection")
print("  Facts are injected INTO model weights")
print("  No RAG. No prompt injection. True memory.")
print("=" * 50)

# Check if we have a trained encoder to load config from
checkpoint = None
checkpoint_config = None
if args.encoder and os.path.exists(args.encoder):
    print(f"\nLoading checkpoint config from: {args.encoder}")
    checkpoint = torch.load(args.encoder, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    print(f"  Checkpoint config: {checkpoint_config}")

# Create Mneme config (use checkpoint config if available)
if checkpoint_config:
    MNEME_CONFIG = MnemeConfig(
        delta_rank=checkpoint_config.get("delta_rank", 16),
        target_layers=checkpoint_config.get("target_layers", [4, 8, 12, 16, 20, 24]),
        encoder_hidden_size=checkpoint_config.get("encoder_hidden_size", 768),
        encoder_layers=checkpoint_config.get("encoder_layers", 4),
        max_memories_active=64,
        memory_path="mneme_memories"
    )
else:
    # Default to high-quality config (matches train_vastai.py)
    MNEME_CONFIG = MnemeConfig(
        delta_rank=16,
        target_layers=[4, 8, 12, 16, 20, 24],
        encoder_hidden_size=768,
        encoder_layers=4,
        max_memories_active=64,
        memory_path="mneme_memories"
    )

print("\nLoading Qwen3-4B...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Create Mneme model
print("\nInitializing Mneme...")
model = MnemeModel(base_model, tokenizer, MNEME_CONFIG)

# Load trained encoder if checkpoint was loaded
if checkpoint:
    model.encoder.load_state_dict(checkpoint["encoder_state"])
    print(f"Loaded trained encoder from: {args.encoder}")
else:
    print("Using untrained encoder (run train_vastai.py first for best results)")

# Refresh memories (apply any existing memories to weights)
model.refresh_memories()

print(f"\nReady on {device}")
print(f"Active memories: {len(model.memory_bank)}")
print("\nCommands: /remember, /memories, /clear, /stats, /help, /quit\n")

# Chat history
history = []


def extract_facts(text):
    """Extract facts from user message and convert to third-person."""
    facts = []

    patterns = [
        # Name patterns
        (r"(?:my name is|i'm called|call me|i am)\s+([A-Z][a-z]+)", "The user's name is {}"),
        (r"(?:i'm|im|i am)\s+([A-Z][a-z]+)(?:\s|$|,|\.)", "The user's name is {}"),

        # Age patterns
        (r"(?:i'm|im|i am)\s+(\d{1,3})\s*(?:years? old|yrs?)?", "The user is {} years old"),
        (r"(?:my age is|age:?)\s*(\d{1,3})", "The user is {} years old"),

        # Location patterns
        (r"(?:i live in|i'm from|im from|i am from|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "The user lives in {}"),
        (r"(?:i'm in|im in|i am in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "The user is in {}"),

        # Preferences
        (r"(?:i like|i love|i enjoy)\s+(.+?)(?:\.|,|$)", "The user likes {}"),
        (r"(?:my favorite|my favourite)\s+(\w+)\s+is\s+(.+?)(?:\.|,|$)", "The user's favorite {} is {}"),

        # Occupation
        (r"(?:i'm a|im a|i am a|i work as a?)\s+([a-z]+(?:\s+[a-z]+)?)", "The user is a {}"),
    ]

    text_lower = text.lower()

    for pattern, template in patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            if len(groups) == 1:
                fact = template.format(groups[0].strip().title())
            elif len(groups) == 2:
                fact = template.format(groups[0].strip(), groups[1].strip())
            else:
                continue

            if fact not in facts:
                facts.append(fact)

    return facts


def chat(user_input):
    """Send a message and get a response."""
    global history

    history.append({"role": "user", "content": user_input})

    # Auto-extract and inject facts
    extracted_facts = extract_facts(user_input)
    for fact in extracted_facts:
        mem_id = model.inject_memory(fact)
        print(f"[Memory injected: {fact}]")

    # Build messages (NO memory injection into prompt - it's in the weights!)
    messages = []

    # Simple system prompt - memory is IN the weights, not the prompt
    messages.append({
        "role": "system",
        "content": "You are a helpful AI assistant. Be friendly and natural in conversation."
    })

    # Add conversation history
    messages.extend(history[-10:])  # Last 10 turns

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            streamer=streamer,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": response})
    print()


def handle_command(cmd):
    """Handle special commands."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command in ['/quit', '/exit', '/q']:
        return 'quit'

    elif command == '/remember':
        if not args:
            print("Usage: /remember <fact to inject into weights>")
            print("Example: /remember The user's name is Alice")
        else:
            mem_id = model.inject_memory(args)
            print(f"Injected into weights: \"{args}\"")
            print(f"Memory ID: {mem_id}")
            print(f"Total memories: {len(model.memory_bank)}")
        return 'continue'

    elif command == '/memories':
        memories = model.list_memories()
        if memories:
            print("\n=== Memories (Injected into Weights) ===")
            for m in memories:
                print(f"  [{m['id']}] {m['text']}")
            print(f"\nTotal: {len(memories)} memories")
            print("=" * 40 + "\n")
        else:
            print("No memories stored.\n")
        return 'continue'

    elif command == '/clear':
        model.clear_memories()
        print("All memories cleared. Weights restored to original.\n")
        return 'continue'

    elif command == '/stats':
        stats = model.get_stats()
        print("\n=== Mneme Statistics ===")
        print(f"  Memories: {stats['num_memories']}")
        print(f"  Target layers: {stats['target_layers']}")
        print(f"  Delta rank: {stats['delta_rank']}")
        print(f"  Storage: {stats['memory_path']}")
        print("=" * 24 + "\n")
        return 'continue'

    elif command == '/help':
        print("\n=== Mneme Commands ===")
        print("  /remember <fact>  - Inject fact into model weights")
        print("  /memories         - List all injected memories")
        print("  /clear            - Clear all memories, restore weights")
        print("  /stats            - Show memory statistics")
        print("  /quit             - Exit")
        print()
        print("Facts are automatically extracted from conversation.")
        print("Memory is stored IN THE WEIGHTS, not as text!")
        print("=" * 22 + "\n")
        return 'continue'

    elif command.startswith('/'):
        print(f"Unknown command: {command}. Type /help for commands.\n")
        return 'continue'

    return None


# Main loop
if __name__ == "__main__":
    print("Mneme Chatbot")
    print("Type /help for commands\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Check for commands
            cmd_result = handle_command(user_input)
            if cmd_result == 'quit':
                break
            elif cmd_result == 'continue':
                continue

            # Regular chat
            print("Assistant: ", end="", flush=True)
            chat(user_input)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")

    finally:
        print("Goodbye!")
