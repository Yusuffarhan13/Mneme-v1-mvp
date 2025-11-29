#!/usr/bin/env python3
"""
Upload trained model to Hugging Face Hub.

Usage:
    python upload_to_hf.py
    python upload_to_hf.py --model ./checkpoints_rtx6000/final --repo your-username/model-name
"""

import argparse
import os
from pathlib import Path


def upload_model(model_path: str, repo_id: str, private: bool = False):
    """Upload model to Hugging Face Hub."""

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import HfApi, login

    # Check if model path exists
    if not Path(model_path).exists():
        print(f"Error: Model path '{model_path}' does not exist!")
        print("\nAvailable checkpoints:")
        for p in Path(".").glob("checkpoints*"):
            print(f"  {p}")
        return False

    print(f"\nUploading model to Hugging Face Hub")
    print(f"  Model path: {model_path}")
    print(f"  Repository: {repo_id}")
    print(f"  Private: {private}")
    print()

    # Login
    print("Logging in to Hugging Face...")
    try:
        login()
    except Exception as e:
        print(f"Login failed: {e}")
        print("\nTo login, run: huggingface-cli login")
        print("Or set HF_TOKEN environment variable")
        return False

    # Create API
    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating repository '{repo_id}'...")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
        print("  Repository ready!")
    except Exception as e:
        print(f"  Warning: {e}")

    # Upload folder
    print(f"\nUploading files from '{model_path}'...")
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Coconut latent thinking model"
        )
        print("\nUpload complete!")
        print(f"\nModel available at: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="./checkpoints_rtx6000/final",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--repo", "-r",
        type=str,
        default="Yusuffarhan13/qwen-coconut-latent",
        help="Hugging Face repo ID (username/model-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    args = parser.parse_args()

    success = upload_model(
        model_path=args.model,
        repo_id=args.repo,
        private=args.private
    )

    if success:
        print("\n" + "=" * 50)
        print("SUCCESS! Your model is now on Hugging Face.")
        print("=" * 50)
        print(f"\nTo download and use:")
        print(f"  git lfs install")
        print(f"  git clone https://huggingface.co/{args.repo}")
        print()
    else:
        print("\nUpload failed. Please check errors above.")


if __name__ == "__main__":
    main()
