# Mneme: Neural Episodic Weight Injection

A breakthrough memory system that injects facts directly INTO model weights. No RAG. No prompt injection. True neural memory.

## What Makes This Different

| Feature | Traditional RAG | MemoryLLM | **Mneme** |
|---------|----------------|-----------|-----------|
| Mechanism | Vector search + prompt | Attention pool | **Weight injection** |
| Memory Location | External DB | Fixed pool | **In the weights** |
| Capacity | Limited by context | Fixed | **Unlimited** |
| Retrieval | Similarity search | Soft attention | **Hard encoding** |

## Architecture

```
Fact Text → MemoryEncoder (Hypernetwork) → Δw (low-rank) → Base Weights + Δw → Output
```

- **MemoryEncoder**: Transformer that maps text → weight deltas
- **Low-Rank Deltas**: LoRA-style `Δw = A @ B.T` (tiny ~1KB per fact)
- **Target Layers**: MLP projections at layers [4, 8, 12, 16, 20, 24]

## Training on Vast.ai (Recommended)

### 1. Rent a GPU

Go to [vast.ai](https://vast.ai) and rent an instance with:
- **GPU**: RTX 6000 Pro (48GB) or better
- **For best results**: H100 or A100 (80GB+)
- **Disk**: 50GB+
- **Image**: PyTorch 2.0+ with CUDA

### 2. Connect and Setup

```bash
# SSH into your instance
ssh -p <PORT> root@<IP>

# Clone the repo
git clone https://github.com/YOUR_USERNAME/mneme.git
cd mneme

# Run setup
chmod +x setup_vastai.sh
./setup_vastai.sh
```

### 3. Train (1 hour for perfect quality)

```bash
# For 96GB VRAM (H100/A100):
python train_vastai.py --epochs 200 --batch-size 32 --rank 16

# For 48GB VRAM (RTX 6000):
python train_vastai.py --epochs 200 --batch-size 16 --rank 16

# For 24GB VRAM (RTX 3090/4090):
python train_vastai.py --epochs 200 --batch-size 8 --rank 8
```

### 4. Download Trained Model

```bash
# On Vast.ai
cd mneme_trained
tar -czvf trained_model.tar.gz *.pt

# On your local machine
scp -P <PORT> root@<IP>:~/mneme/mneme_trained/trained_model.tar.gz .
```

## Local Usage

### Test the Chatbot

```bash
python qwen.py --encoder mneme_trained/best_encoder.pt
```

### Commands

- `/remember <fact>` - Inject a fact into weights
- `/memories` - List all injected memories
- `/clear` - Clear all memories
- `/stats` - Show memory statistics
- `/quit` - Exit

### Example Session

```
You: /remember My name is Yusuf
Injected into weights: "My name is Yusuf"

You: /remember I work at Google
Injected into weights: "I work at Google"

You: What is my name?
Assistant: Your name is Yusuf.

You: Where do I work?
Assistant: You work at Google.
```

## File Structure

```
mneme/
├── mneme.py              # Core architecture (MemoryEncoder, MnemeModel)
├── qwen.py               # Chatbot interface
├── train_vastai.py       # High-quality training script
├── train_mneme.py        # Basic training script
├── setup_vastai.sh       # Vast.ai setup
├── requirements.txt      # Dependencies
└── mneme_trained/        # Saved checkpoints
    ├── best_encoder.pt
    └── final_encoder.pt
```

## Technical Details

### Configuration

```python
HighQualityMnemeConfig:
    hidden_size: 2560          # Qwen3-4B
    intermediate_size: 9728    # MLP size
    encoder_hidden_size: 768   # Larger encoder
    encoder_layers: 4          # Deeper encoder
    delta_rank: 16             # More expressive deltas
    target_layers: [4,8,12,16,20,24]  # 6 injection points
```

### Training

- **Dataset**: 5000+ fact/question/answer triplets
- **Epochs**: 200 for convergence
- **Batch Size**: 32 (with 96GB VRAM)
- **LR**: 5e-4 with cosine annealing warm restarts
- **Mixed Precision**: BF16 for speed

### Memory Efficiency

Each fact requires only ~1KB of storage (low-rank deltas).
- Rank 4: ~12KB per memory
- Rank 16: ~48KB per memory (recommended)

## Research Notes

This implements a hypernetwork approach to episodic memory:

1. **Hypernetwork**: Small network that generates weights for a larger network
2. **Low-Rank**: Inspired by LoRA, we use `A @ B.T` factorization
3. **Forward Hooks**: Apply deltas during forward pass for gradient flow
4. **No Forgetting**: Each memory is stored independently, additive composition

Unlike MemoryLLM which uses attention over a fixed memory pool, Mneme directly modifies the model's weights to encode facts.

## Citation

```bibtex
@software{mneme2024,
  title={Mneme: Neural Episodic Weight Injection},
  year={2024},
  description={Direct weight modification for LLM memory}
}
```

## License

MIT
