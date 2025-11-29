"""
Coconut Model Wrapper
Wraps Qwen for latent space reasoning with <bot> and <eot> token handling.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class CoconutQwen(nn.Module):
    """
    Wrapper for Qwen that enables latent space reasoning.

    Key features:
    - Handles <bot> tokens for latent thinking
    - Handles <eot> tokens for stopping latent mode
    - Supports QLoRA training
    - Implements loss masking for <bot> tokens
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        use_qlora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        device: str = "cuda"
    ):
        super().__init__()

        self.model_id = model_id
        self.device = device
        self.use_qlora = use_qlora

        # Token IDs (set after tokenizer initialization)
        self.bot_token_id: Optional[int] = None
        self.eot_token_id: Optional[int] = None

        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._add_special_tokens()

        # Load model with quantization if using QLoRA
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"
            )
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)

            # Apply LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa"
            )

        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Get embedding layer reference
        self.embed_tokens = self.model.get_input_embeddings()

    def _add_special_tokens(self):
        """Add <bot> and <eot> tokens to tokenizer."""
        special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens")

        self.bot_token_id = self.tokenizer.convert_tokens_to_ids("<bot>")
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<eot>")
        print(f"Token IDs: <bot>={self.bot_token_id}, <eot>={self.eot_token_id}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with optional latent mode handling.

        Args:
            input_ids: Token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            inputs_embeds: Input embeddings (for latent mode) [batch, seq, hidden]
            labels: Training labels [batch, seq]
            output_hidden_states: Return hidden states
            return_dict: Return as dict

        Returns:
            Model outputs with optional loss masking for <bot> tokens
        """
        # If labels provided, mask loss on <bot> tokens
        if labels is not None and input_ids is not None:
            labels = self._mask_bot_tokens(input_ids, labels)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def _mask_bot_tokens(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Mask loss on <bot> tokens - we don't want gradient on these.
        Only train on final answer prediction.

        Args:
            input_ids: Token IDs
            labels: Training labels

        Returns:
            Modified labels with -100 at <bot> positions
        """
        # Find <bot> token positions
        bot_mask = (input_ids == self.bot_token_id)

        # Set labels to -100 at <bot> positions (ignored in loss)
        labels = labels.clone()
        labels[bot_mask] = -100

        return labels

    def latent_thinking_step(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one latent thinking step.
        Uses last hidden state as next input embedding.

        Args:
            inputs_embeds: Current embeddings [batch, seq, hidden]
            attention_mask: Current attention mask [batch, seq]

        Returns:
            New embeddings and attention mask with appended latent thought
        """
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )

        # Get last hidden state from last position
        last_hidden = outputs.hidden_states[-1][:, -1:, :]

        # Match dtype
        if last_hidden.dtype != inputs_embeds.dtype:
            last_hidden = last_hidden.to(inputs_embeds.dtype)

        # Append to sequence
        new_embeds = torch.cat([inputs_embeds, last_hidden], dim=1)

        # Extend attention mask
        new_mask = torch.cat([
            attention_mask,
            torch.ones((attention_mask.size(0), 1),
                       device=attention_mask.device,
                       dtype=attention_mask.dtype)
        ], dim=1)

        return new_embeds, new_mask

    def generate_with_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_latent_steps: int = 10,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate with fixed number of latent thinking steps.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            num_latent_steps: Number of latent thinking steps
            max_new_tokens: Maximum tokens to generate after thinking
            temperature: Sampling temperature
            top_p: Top-p sampling
            do_sample: Whether to sample

        Returns:
            Generated token IDs
        """
        # Get initial embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Perform latent thinking
        for _ in range(num_latent_steps):
            inputs_embeds, attention_mask = self.latent_thinking_step(
                inputs_embeds, attention_mask
            )

        # Add <eot> token to signal end of thinking
        eot_embed = self.embed_tokens(
            torch.tensor([[self.eot_token_id]], device=input_ids.device)
        )
        if eot_embed.dtype != inputs_embeds.dtype:
            eot_embed = eot_embed.to(inputs_embeds.dtype)

        inputs_embeds = torch.cat([inputs_embeds, eot_embed], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=1)

        # Generate answer
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

        return outputs

    def generate_adaptive(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_latent_steps: int = 20,
        confidence_threshold: float = 0.8,
        max_new_tokens: int = 256,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate with adaptive latent thinking - model decides when to stop.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_latent_steps: Maximum latent steps
            confidence_threshold: Entropy threshold for stopping
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated tokens and number of latent steps used
        """
        inputs_embeds = self.embed_tokens(input_ids)
        steps_used = 0

        for step in range(max_latent_steps):
            # Get outputs for current sequence
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            last_hidden = outputs.hidden_states[-1][:, -1:, :]

            # Check if model wants to output <eot> (stop thinking)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            # Get probability of <eot> token
            eot_prob = probs[:, self.eot_token_id].item()

            if eot_prob > confidence_threshold:
                # Model is confident - stop thinking
                break

            # Continue thinking - append hidden state
            if last_hidden.dtype != inputs_embeds.dtype:
                last_hidden = last_hidden.to(inputs_embeds.dtype)

            inputs_embeds = torch.cat([inputs_embeds, last_hidden], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            ], dim=1)

            steps_used = step + 1

        # Add <eot> and generate
        eot_embed = self.embed_tokens(
            torch.tensor([[self.eot_token_id]], device=input_ids.device)
        )
        if eot_embed.dtype != inputs_embeds.dtype:
            eot_embed = eot_embed.to(inputs_embeds.dtype)

        inputs_embeds = torch.cat([inputs_embeds, eot_embed], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=1)

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        return outputs, steps_used

    def save_pretrained(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load saved model."""
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)

        instance.bot_token_id = instance.tokenizer.convert_tokens_to_ids("<bot>")
        instance.eot_token_id = instance.tokenizer.convert_tokens_to_ids("<eot>")
        instance.embed_tokens = instance.model.get_input_embeddings()

        return instance


class CoconutTrainer:
    """
    Custom trainer for Coconut curriculum learning.
    Handles stage transitions and loss masking.
    """

    def __init__(
        self,
        model: CoconutQwen,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        warmup_steps: int = 100,
        max_length: int = 512
    ):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_length = max_length

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

    def tokenize_example(self, example: Dict) -> Dict[str, torch.Tensor]:
        """Tokenize a single training example."""
        # Format: input + output
        full_text = f"{example['input']} {example['output']}"

        tokens = self.model.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Create labels (shift by 1 for next-token prediction)
        labels = tokens["input_ids"].clone()

        # Mask padding tokens
        labels[labels == self.model.tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Shuffle data
        import random
        random.shuffle(self.train_data)

        self.optimizer.zero_grad()

        for i, example in enumerate(self.train_data):
            # Tokenize
            batch = self.tokenize_example(example)

            # Move to device
            input_ids = batch["input_ids"].unsqueeze(0).to(self.model.device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(self.model.device)
            labels = batch["labels"].unsqueeze(0).to(self.model.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()

            total_loss += outputs.loss.item()
            num_batches += 1

            # Gradient accumulation
            if (i + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Progress
            if (i + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Step {i+1}/{len(self.train_data)}, Loss: {avg_loss:.4f}")

        return total_loss / num_batches

    def evaluate(self) -> float:
        """Evaluate on eval data."""
        if not self.eval_data:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_examples = 0

        with torch.no_grad():
            for example in self.eval_data:
                batch = self.tokenize_example(example)
                input_ids = batch["input_ids"].unsqueeze(0).to(self.model.device)
                attention_mask = batch["attention_mask"].unsqueeze(0).to(self.model.device)
                labels = batch["labels"].unsqueeze(0).to(self.model.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                num_examples += 1

        return total_loss / num_examples


if __name__ == "__main__":
    # Test model loading
    print("Testing CoconutQwen...")

    model = CoconutQwen(use_qlora=True)
    print(f"\nModel loaded successfully!")
    print(f"<bot> token ID: {model.bot_token_id}")
    print(f"<eot> token ID: {model.eot_token_id}")

    # Test tokenization
    test_text = "What is 2+2? <bot> <bot> <bot> <eot> 4"
    tokens = model.tokenizer(test_text, return_tensors="pt")
    print(f"\nTest tokenization:")
    print(f"  Text: {test_text}")
    print(f"  Tokens: {tokens['input_ids']}")
    print(f"  Decoded: {model.tokenizer.decode(tokens['input_ids'][0])}")
