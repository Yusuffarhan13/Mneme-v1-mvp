"""
Upload trained Mneme encoder to Hugging Face Hub

Usage:
    python upload_to_hf.py --token YOUR_HF_TOKEN --repo your-username/mneme-encoder
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo, upload_file

def main():
    parser = argparse.ArgumentParser(description="Upload Mneme encoder to HuggingFace")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--repo", type=str, required=True, help="Repo name (e.g., username/mneme-encoder)")
    parser.add_argument("--checkpoint", type=str, default="mneme_trained/best_encoder.pt",
                        help="Path to checkpoint")
    args = parser.parse_args()

    print(f"Uploading to: {args.repo}")

    # Create repo if it doesn't exist
    api = HfApi()
    try:
        create_repo(args.repo, token=args.token, repo_type="model", exist_ok=True)
        print(f"Created/verified repo: {args.repo}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload checkpoint
    print(f"Uploading {args.checkpoint}...")
    api.upload_file(
        path_or_fileobj=args.checkpoint,
        path_in_repo="best_encoder.pt",
        repo_id=args.repo,
        token=args.token,
    )
    print("Uploaded: best_encoder.pt")

    # Create model card
    model_card = """---
license: mit
tags:
- mneme
- memory
- weight-injection
- qwen
---

# Mneme: Neural Episodic Weight Injection Encoder

Trained encoder for the Mneme memory system - injects facts directly into LLM weights.

## Usage

```bash
# Clone the repo
git clone https://github.com/Yusuffarhan13/Mneme-v1-mvp.git
cd Mneme-v1-mvp

# Download the encoder
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='YOUR_REPO', filename='best_encoder.pt', local_dir='mneme_trained')"

# Run
python qwen.py --encoder mneme_trained/best_encoder.pt
```

## Training Config

- **Delta rank**: 16
- **Target layers**: [4, 8, 12, 16, 20, 24]
- **Encoder**: 768 hidden, 4 layers
- **Base model**: Qwen/Qwen3-4B

## What This Does

Injects facts directly INTO model weights (no RAG, no prompt injection):

```
/remember My name is Yusuf
/remember I work at Google
What is my name?  →  "Your name is Yusuf"
Where do I work?  →  "You work at Google"
```
"""

    # Save and upload model card
    with open("/tmp/README.md", "w") as f:
        f.write(model_card.replace("YOUR_REPO", args.repo))

    api.upload_file(
        path_or_fileobj="/tmp/README.md",
        path_in_repo="README.md",
        repo_id=args.repo,
        token=args.token,
    )
    print("Uploaded: README.md")

    print(f"\n✅ Done! Your model is at: https://huggingface.co/{args.repo}")
    print(f"\nTo download on your local PC:")
    print(f"  pip install huggingface_hub")
    print(f"  python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='{args.repo}', filename='best_encoder.pt', local_dir='mneme_trained')\"")

if __name__ == "__main__":
    main()
