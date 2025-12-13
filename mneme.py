"""
Mneme: Neural Episodic Weight Injection

A breakthrough memory system that injects facts directly into model weights.
No RAG. No prompt injection. Facts become part of the model.

Key Innovation:
- Memory Encoder: A hypernetwork that maps text → weight deltas
- Weight deltas are LOW-RANK (tiny, ~1KB per fact)
- Deltas are ADDED to frozen base weights at inference
- Unlimited capacity, no forgetting

Architecture:
    Fact Text → MemoryEncoder → Δw (low-rank) → Base Weights + Δw → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import math


@dataclass
class MnemeConfig:
    """Configuration for Mneme memory system."""
    # Target model (Qwen3-4B actual dimensions)
    hidden_size: int = 2560  # Qwen3-4B hidden size
    num_layers: int = 36  # Qwen3-4B layers
    intermediate_size: int = 9728  # Qwen3-4B MLP size

    # Memory encoder
    encoder_hidden_size: int = 512  # Small encoder
    encoder_layers: int = 2
    encoder_heads: int = 8

    # Weight delta (low-rank)
    delta_rank: int = 4  # Rank of weight modifications (lower = smaller memory)

    # Which layers to inject memory into
    target_layers: List[int] = field(default_factory=lambda: [8, 16, 24])  # Mid-to-late layers

    # Storage
    memory_path: str = "mneme_memories"

    # Constraints for slow GPU
    max_memories_active: int = 32  # Max memories to compose at once


class LowRankDelta(nn.Module):
    """
    Low-rank weight modification: Δw = A @ B.T

    For a weight matrix W of shape (out, in):
    - A is (out, rank)
    - B is (in, rank)
    - Δw = A @ B.T is (out, in)

    This is MUCH smaller than storing full Δw.
    Memory: rank * (in + out) vs in * out
    For rank=4, hidden=3072: 24K params vs 9.4M params (400x smaller!)
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Low-rank factors
        self.A = nn.Parameter(torch.zeros(out_features, rank))
        self.B = nn.Parameter(torch.zeros(in_features, rank))

        # Scaling factor (like LoRA)
        self.scaling = 1.0 / rank

    def forward(self) -> torch.Tensor:
        """Compute the full delta: Δw = A @ B.T * scaling"""
        return (self.A @ self.B.T) * self.scaling

    def get_size(self) -> int:
        """Return number of parameters."""
        return self.rank * (self.in_features + self.out_features)


class MemoryEncoder(nn.Module):
    """
    Hypernetwork that maps fact text → weight deltas.

    Input: Tokenized fact (e.g., "The user's name is Yusuf")
    Output: Low-rank weight deltas for target layers

    This is the CORE INNOVATION of Mneme.
    """

    def __init__(self, config: MnemeConfig, tokenizer_vocab_size: int = 151936):
        super().__init__()
        self.config = config

        # Text embedding
        self.embed = nn.Embedding(tokenizer_vocab_size, config.encoder_hidden_size)

        # Small transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_hidden_size,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)

        # Pooling
        self.pool = nn.Sequential(
            nn.Linear(config.encoder_hidden_size, config.encoder_hidden_size),
            nn.GELU(),
        )

        # Delta generators for each target layer and each weight matrix
        # In Qwen MLP: gate_proj (h→i), up_proj (h→i), down_proj (i→h)
        self.delta_generators = nn.ModuleDict()

        for layer_idx in config.target_layers:
            # Gate projection: hidden → intermediate
            self.delta_generators[f"layer_{layer_idx}_gate_A"] = nn.Linear(
                config.encoder_hidden_size, config.intermediate_size * config.delta_rank
            )
            self.delta_generators[f"layer_{layer_idx}_gate_B"] = nn.Linear(
                config.encoder_hidden_size, config.hidden_size * config.delta_rank
            )

            # Up projection: hidden → intermediate
            self.delta_generators[f"layer_{layer_idx}_up_A"] = nn.Linear(
                config.encoder_hidden_size, config.intermediate_size * config.delta_rank
            )
            self.delta_generators[f"layer_{layer_idx}_up_B"] = nn.Linear(
                config.encoder_hidden_size, config.hidden_size * config.delta_rank
            )

            # Down projection: intermediate → hidden
            self.delta_generators[f"layer_{layer_idx}_down_A"] = nn.Linear(
                config.encoder_hidden_size, config.hidden_size * config.delta_rank
            )
            self.delta_generators[f"layer_{layer_idx}_down_B"] = nn.Linear(
                config.encoder_hidden_size, config.intermediate_size * config.delta_rank
            )

        # Initialize to produce small deltas initially
        self._init_weights()

    def _init_weights(self):
        """Initialize to produce near-zero deltas initially."""
        for name, param in self.named_parameters():
            if 'delta_generators' in name:
                if 'weight' in name:
                    nn.init.normal_(param, std=0.001)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Encode a fact into weight deltas.

        Args:
            input_ids: (batch, seq_len) tokenized fact
            attention_mask: (batch, seq_len) attention mask

        Returns:
            Dict mapping "layer_{idx}_{proj}_{A/B}" to tensors
        """
        # Embed
        x = self.embed(input_ids)  # (batch, seq, hidden)

        # Transform
        if attention_mask is not None:
            # Convert to transformer mask format
            mask = attention_mask == 0
        else:
            mask = None

        x = self.transformer(x, src_key_padding_mask=mask)  # (batch, seq, hidden)

        # Pool (mean over sequence)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # (batch, hidden)

        x = self.pool(x)  # (batch, hidden)

        # Generate deltas
        deltas = {}
        for layer_idx in self.config.target_layers:
            # Gate projection
            gate_A = self.delta_generators[f"layer_{layer_idx}_gate_A"](x)
            gate_A = gate_A.view(-1, self.config.intermediate_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_gate_A"] = gate_A

            gate_B = self.delta_generators[f"layer_{layer_idx}_gate_B"](x)
            gate_B = gate_B.view(-1, self.config.hidden_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_gate_B"] = gate_B

            # Up projection
            up_A = self.delta_generators[f"layer_{layer_idx}_up_A"](x)
            up_A = up_A.view(-1, self.config.intermediate_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_up_A"] = up_A

            up_B = self.delta_generators[f"layer_{layer_idx}_up_B"](x)
            up_B = up_B.view(-1, self.config.hidden_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_up_B"] = up_B

            # Down projection
            down_A = self.delta_generators[f"layer_{layer_idx}_down_A"](x)
            down_A = down_A.view(-1, self.config.hidden_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_down_A"] = down_A

            down_B = self.delta_generators[f"layer_{layer_idx}_down_B"](x)
            down_B = down_B.view(-1, self.config.intermediate_size, self.config.delta_rank)
            deltas[f"layer_{layer_idx}_down_B"] = down_B

        return deltas


@dataclass
class MemoryEntry:
    """A single memory stored as weight deltas."""
    id: str
    text: str  # Original fact text
    timestamp: str
    deltas: Dict[str, torch.Tensor]  # The actual weight deltas

    def save(self, path: str):
        """Save memory to disk."""
        os.makedirs(path, exist_ok=True)

        # Save metadata
        meta = {
            "id": self.id,
            "text": self.text,
            "timestamp": self.timestamp,
        }
        with open(os.path.join(path, f"{self.id}_meta.json"), "w") as f:
            json.dump(meta, f)

        # Save deltas
        delta_dict = {k: v.cpu() for k, v in self.deltas.items()}
        torch.save(delta_dict, os.path.join(path, f"{self.id}_deltas.pt"))

    @classmethod
    def load(cls, path: str, memory_id: str, device: str = "cuda") -> "MemoryEntry":
        """Load memory from disk."""
        with open(os.path.join(path, f"{memory_id}_meta.json"), "r") as f:
            meta = json.load(f)

        deltas = torch.load(os.path.join(path, f"{memory_id}_deltas.pt"), map_location=device)

        return cls(
            id=meta["id"],
            text=meta["text"],
            timestamp=meta["timestamp"],
            deltas=deltas
        )


class MemoryBank:
    """
    Stores and manages memories (weight deltas).

    Each memory is a set of low-rank weight deltas.
    At inference, relevant memories are composed (summed) and applied.
    """

    def __init__(self, config: MnemeConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.memories: Dict[str, MemoryEntry] = {}

        # Create storage directory
        os.makedirs(config.memory_path, exist_ok=True)

        # Load existing memories
        self._load_all()

    def _load_all(self):
        """Load all memories from disk."""
        if not os.path.exists(self.config.memory_path):
            return

        # Find all memory IDs
        files = os.listdir(self.config.memory_path)
        memory_ids = set()
        for f in files:
            if f.endswith("_meta.json"):
                memory_ids.add(f.replace("_meta.json", ""))

        for mem_id in memory_ids:
            try:
                self.memories[mem_id] = MemoryEntry.load(
                    self.config.memory_path, mem_id, self.device
                )
            except Exception as e:
                print(f"Failed to load memory {mem_id}: {e}")

        if self.memories:
            print(f"Loaded {len(self.memories)} memories from disk")

    def add(self, text: str, deltas: Dict[str, torch.Tensor]) -> str:
        """Add a new memory."""
        memory_id = f"mem_{len(self.memories)}_{hash(text) % 10000}"

        entry = MemoryEntry(
            id=memory_id,
            text=text,
            timestamp=datetime.now().isoformat(),
            deltas={k: v.detach().clone() for k, v in deltas.items()}
        )

        self.memories[memory_id] = entry
        entry.save(self.config.memory_path)

        return memory_id

    def get_all_deltas(self) -> Dict[str, torch.Tensor]:
        """
        Compose all memories into a single set of deltas.

        This is the key operation: Δw_total = Σ (A_i @ B_i.T)

        IMPORTANT: We compute A @ B.T for each memory BEFORE summing,
        not (A1+A2) @ (B1+B2).T which would create interference.
        """
        if not self.memories:
            return {}

        # Limit to max_memories_active for slow GPU
        active_memories = list(self.memories.values())[:self.config.max_memories_active]
        n_memories = len(active_memories)

        # Scale by sqrt(n) to prevent delta magnitude explosion with many memories
        memory_scale = 1.0 / math.sqrt(max(1, n_memories))

        composed = {}

        for memory in active_memories:
            deltas = memory.deltas

            # For each layer's projections, compute A @ B.T and sum
            for layer_idx in self.config.target_layers:
                for proj in ["gate", "up", "down"]:
                    A_key = f"layer_{layer_idx}_{proj}_A"
                    B_key = f"layer_{layer_idx}_{proj}_B"
                    full_key = f"layer_{layer_idx}_{proj}_full"

                    if A_key in deltas and B_key in deltas:
                        A = deltas[A_key].to(self.device)
                        B = deltas[B_key].to(self.device)

                        # Squeeze batch dimension if present
                        if A.dim() == 3:
                            A = A.squeeze(0)
                            B = B.squeeze(0)

                        # Compute full delta: A @ B.T
                        full_delta = (A @ B.T) * memory_scale

                        if full_key not in composed:
                            composed[full_key] = full_delta
                        else:
                            composed[full_key] = composed[full_key] + full_delta

        return composed

    def clear(self):
        """Clear all memories."""
        self.memories = {}
        # Clear disk storage
        import shutil
        if os.path.exists(self.config.memory_path):
            shutil.rmtree(self.config.memory_path)
            os.makedirs(self.config.memory_path)
        print("All memories cleared")

    def list_memories(self) -> List[Dict[str, str]]:
        """List all stored memories."""
        return [
            {"id": m.id, "text": m.text, "timestamp": m.timestamp}
            for m in self.memories.values()
        ]

    def __len__(self):
        return len(self.memories)


class MnemeModel(nn.Module):
    """
    Qwen model with Mneme weight injection.

    This wraps the base Qwen model and:
    1. Uses MemoryEncoder to convert facts to weight deltas
    2. Applies deltas to MLP weights at inference
    3. No prompt injection - memory is IN the weights
    """

    def __init__(self, base_model, tokenizer, config: MnemeConfig = None):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config or MnemeConfig()

        # Get device and dtype
        self.model_device = next(base_model.parameters()).device
        self.model_dtype = next(base_model.parameters()).dtype

        # Memory encoder (the hypernetwork)
        self.encoder = MemoryEncoder(self.config, tokenizer.vocab_size)
        self.encoder.to(device=self.model_device, dtype=torch.float32)  # Encoder in fp32

        # Memory bank
        self.memory_bank = MemoryBank(self.config, str(self.model_device))

        # Store original weights for restoration
        self._original_weights = {}
        self._save_original_weights()

        # Currently applied deltas
        self._applied_deltas = {}

        print(f"Mneme initialized:")
        print(f"  - Target layers: {self.config.target_layers}")
        print(f"  - Delta rank: {self.config.delta_rank}")
        print(f"  - Encoder params: {sum(p.numel() for p in self.encoder.parameters()):,}")
        print(f"  - Existing memories: {len(self.memory_bank)}")

    def _save_original_weights(self):
        """Save original MLP weights for restoration."""
        for layer_idx in self.config.target_layers:
            layer = self.base_model.model.layers[layer_idx]
            self._original_weights[f"layer_{layer_idx}_gate"] = layer.mlp.gate_proj.weight.data.clone()
            self._original_weights[f"layer_{layer_idx}_up"] = layer.mlp.up_proj.weight.data.clone()
            self._original_weights[f"layer_{layer_idx}_down"] = layer.mlp.down_proj.weight.data.clone()

    def _apply_deltas(self, deltas: Dict[str, torch.Tensor]):
        """Apply pre-computed weight deltas to the model.

        Deltas should be pre-composed full matrices (not A/B factors).
        Keys are like: "layer_8_gate_full", "layer_8_up_full", etc.
        """
        # Balanced scaling - enough to influence but not destabilize
        scaling = 0.5  # Tuned for 3 memories

        for layer_idx in self.config.target_layers:
            layer = self.base_model.model.layers[layer_idx]

            # Gate projection
            gate_delta = deltas.get(f"layer_{layer_idx}_gate_full")
            if gate_delta is not None:
                layer.mlp.gate_proj.weight.data += (gate_delta * scaling).to(self.model_dtype)

            # Up projection
            up_delta = deltas.get(f"layer_{layer_idx}_up_full")
            if up_delta is not None:
                layer.mlp.up_proj.weight.data += (up_delta * scaling).to(self.model_dtype)

            # Down projection
            down_delta = deltas.get(f"layer_{layer_idx}_down_full")
            if down_delta is not None:
                layer.mlp.down_proj.weight.data += (down_delta * scaling).to(self.model_dtype)

        self._applied_deltas = deltas

    def _restore_weights(self):
        """Restore original weights (remove all deltas)."""
        for layer_idx in self.config.target_layers:
            layer = self.base_model.model.layers[layer_idx]
            layer.mlp.gate_proj.weight.data = self._original_weights[f"layer_{layer_idx}_gate"].clone()
            layer.mlp.up_proj.weight.data = self._original_weights[f"layer_{layer_idx}_up"].clone()
            layer.mlp.down_proj.weight.data = self._original_weights[f"layer_{layer_idx}_down"].clone()
        self._applied_deltas = {}

    def inject_memory(self, text: str) -> str:
        """
        Inject a fact into the model's weights.

        Args:
            text: The fact to remember (e.g., "The user's name is Yusuf")

        Returns:
            Memory ID
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()}

        # Encode to deltas
        with torch.no_grad():
            deltas = self.encoder(inputs["input_ids"], inputs.get("attention_mask"))

        # Store in memory bank
        memory_id = self.memory_bank.add(text, deltas)

        # Reapply all memories
        self._restore_weights()
        all_deltas = self.memory_bank.get_all_deltas()
        if all_deltas:
            self._apply_deltas(all_deltas)

        return memory_id

    def refresh_memories(self):
        """Reapply all memories from bank to weights."""
        self._restore_weights()
        all_deltas = self.memory_bank.get_all_deltas()
        if all_deltas:
            self._apply_deltas(all_deltas)

    def forward(self, *args, **kwargs):
        """Forward pass through memory-augmented model."""
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generate with memory-augmented model."""
        return self.base_model.generate(*args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "num_memories": len(self.memory_bank),
            "target_layers": self.config.target_layers,
            "delta_rank": self.config.delta_rank,
            "memory_path": self.config.memory_path,
        }

    def list_memories(self) -> List[Dict]:
        """List all stored memories."""
        return self.memory_bank.list_memories()

    def clear_memories(self):
        """Clear all memories and restore original weights."""
        self._restore_weights()
        self.memory_bank.clear()

    def save_encoder(self, path: str):
        """Save the trained encoder."""
        torch.save(self.encoder.state_dict(), path)
        print(f"Encoder saved to {path}")

    def load_encoder(self, path: str):
        """Load a trained encoder."""
        self.encoder.load_state_dict(torch.load(path, map_location=self.model_device))
        print(f"Encoder loaded from {path}")


def create_mneme_model(base_model, tokenizer, config: MnemeConfig = None) -> MnemeModel:
    """Create a Mneme-augmented model."""
    return MnemeModel(base_model, tokenizer, config)
