"""Diagnose why inference is so slow"""

import torch
import subprocess

print("\n" + "="*60)
print("DIAGNOSTIC REPORT")
print("="*60)

# PyTorch info
print("\n1. PyTorch Configuration:")
print(f"   Version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"   Device count: {torch.cuda.device_count()}")

# GPU memory
print("\n2. GPU Memory:")
if torch.cuda.is_available():
    print(f"   Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# nvidia-smi
print("\n3. nvidia-smi:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout)
except:
    print("   nvidia-smi not available")

# Test tensor operations speed
print("\n4. Tensor Operation Speed Test:")
if torch.cuda.is_available():
    import time

    # Warm up
    x = torch.randn(1000, 1000, device='cuda', dtype=torch.bfloat16)
    y = torch.randn(1000, 1000, device='cuda', dtype=torch.bfloat16)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()

    # Test
    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"   100x matmul(1000x1000): {elapsed:.3f}s")
    print(f"   Expected on RTX 4090: <0.1s")
    if elapsed > 0.5:
        print("   ⚠️  WARNING: VERY SLOW! Possible issues:")
        print("      - GPU not being used")
        print("      - Power limit/thermal throttling")
        print("      - Driver issues")
    else:
        print("   ✓ Tensor operations are fast")

print("\n" + "="*60)
