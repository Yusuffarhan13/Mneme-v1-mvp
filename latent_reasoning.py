"""
Latent Space Reasoning for Qwen3-4B
Implements Coconut-style continuous thought reasoning without fine-tuning

Key Concept:
- Instead of generating text tokens during "thinking", reason in hidden state space
- Last hidden state becomes input for next iteration
- After N thinking steps, generate text from enriched context

Features:
- Adaptive thinking: 3-15+ steps based on input complexity
- Works with quantized models (4-bit, fp8, bf16)
- Streaming output for final response
"""

import torch
import re
import time
from typing import Optional, Tuple, List
from transformers import TextStreamer, TextIteratorStreamer
from threading import Thread


class LatentReasoningQwen:
    """
    Latent space reasoning wrapper for Qwen models.

    Implements inference-time latent reasoning by:
    1. Converting input tokens to embeddings
    2. Running N forward passes, accumulating hidden states
    3. Using enriched embeddings for final text generation
    """

    # Keyword sets for complexity estimation
    MATH_KEYWORDS = {
        'calculate', 'compute', 'solve', 'equation', 'formula', 'math',
        'multiply', 'divide', 'add', 'subtract', 'sum', 'product',
        'percentage', 'fraction', 'decimal', 'algebra', 'calculus',
        'times', 'plus', 'minus', 'equals', 'equal',
        '+', '-', '*', '/', '=', '%', '^', 'sqrt', 'root'
    }

    LOGIC_KEYWORDS = {
        'if', 'then', 'therefore', 'because', 'implies', 'hence',
        'conclude', 'deduce', 'infer', 'assume', 'given', 'prove',
        'logic', 'reasoning', 'argument', 'premise', 'conclusion',
        'true', 'false', 'valid', 'invalid', 'contradiction'
    }

    ANALYSIS_KEYWORDS = {
        'compare', 'contrast', 'analyze', 'evaluate', 'pros', 'cons',
        'advantages', 'disadvantages', 'benefits', 'drawbacks',
        'trade-off', 'tradeoff', 'versus', 'vs', 'difference',
        'similarity', 'relationship', 'impact', 'effect', 'cause'
    }

    SIMPLE_PATTERNS = {
        'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye',
        'what is', 'who is', 'when is', 'where is', 'how are you'
    }

    def __init__(
        self,
        router,  # QwenSmartRouter instance
        min_steps: int = 3,
        max_steps: int = 100,  # High limit - model can think as much as needed
        verbose: bool = False
    ):
        """
        Initialize latent reasoning wrapper.

        Args:
            router: QwenSmartRouter instance (handles model loading)
            min_steps: Minimum thinking steps for simple queries
            max_steps: Maximum thinking steps for complex queries (default 100)
            verbose: Print detailed thinking progress
        """
        self.router = router
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.verbose = verbose
        self._fixed_steps = None  # Override for manual step control

    def set_fixed_steps(self, steps: Optional[int]):
        """Set fixed thinking steps (None for adaptive mode)"""
        self._fixed_steps = steps

    def get_current_mode(self) -> str:
        """Get current thinking mode description"""
        if self._fixed_steps is not None:
            return f"Fixed: {self._fixed_steps} steps"
        return f"Adaptive: {self.min_steps}-{self.max_steps} steps"

    def estimate_complexity(self, text: str) -> int:
        """
        Estimate thinking steps needed based on input complexity.

        Args:
            text: User input text

        Returns:
            Recommended number of thinking steps (min_steps to max_steps)
        """
        if self._fixed_steps is not None:
            return self._fixed_steps

        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        # Start with base score
        score = 0

        # Math complexity detection first (before simple patterns)
        math_matches = len(words & self.MATH_KEYWORDS)
        # Also check for numbers and operators in the text
        has_math_expression = bool(re.search(r'\d+\s*[\+\-\*\/\^x×÷]\s*\d+', text, re.IGNORECASE))
        # Check for math-related words
        has_math_words = any(w in text_lower for w in ['calculate', 'compute', 'solve', 'equation', 'math', 'multiply', 'divide'])
        is_math_problem = math_matches > 0 or has_math_expression or has_math_words

        # Check for simple patterns (greetings, basic questions) - but not if it's a math problem
        if not is_math_problem:
            for pattern in self.SIMPLE_PATTERNS:
                if pattern in text_lower:
                    return self.min_steps

        # Math complexity scoring
        if is_math_problem:
            score += 5 + min(math_matches, 4)  # Base score for math problems

        # Logic complexity
        logic_matches = len(words & self.LOGIC_KEYWORDS)
        if logic_matches > 0:
            score += 2 + min(logic_matches, 4)

        # Analysis complexity
        analysis_matches = len(words & self.ANALYSIS_KEYWORDS)
        if analysis_matches > 0:
            score += 2 + min(analysis_matches, 3)

        # Length factor (longer questions often need more thought)
        word_count = len(text.split())
        if word_count > 50:
            score += 3
        elif word_count > 30:
            score += 2
        elif word_count > 15:
            score += 1

        # Nested clauses (commas, semicolons suggest complexity)
        clause_count = text.count(',') + text.count(';') + text.count(':')
        if clause_count > 5:
            score += 2
        elif clause_count > 2:
            score += 1

        # Question marks (multiple questions = more thinking)
        question_count = text.count('?')
        if question_count > 2:
            score += 2
        elif question_count > 1:
            score += 1

        # Map score to steps - scales up for complex queries
        # Score 0-2: min_steps (simple)
        # Score 3-5: 5 steps
        # Score 6-8: 8 steps
        # Score 9-12: 12 steps
        # Score 13-16: 18 steps
        # Score 17-20: 25 steps
        # Score 21+: 35+ steps (very complex)

        if score <= 2:
            steps = self.min_steps
        elif score <= 5:
            steps = 5
        elif score <= 8:
            steps = 8
        elif score <= 12:
            steps = 12
        elif score <= 16:
            steps = 18
        elif score <= 20:
            steps = 25
        else:
            # Very complex - scale linearly
            steps = 35 + (score - 21) * 2

        return min(max(steps, self.min_steps), self.max_steps)

    def _get_model_and_processor(self, messages):
        """Get appropriate model and processor based on message content."""
        if self.router.has_visual_input(messages):
            model, processor = self.router.load_4b_model()
        else:
            model, processor = self.router.load_8b_model()
        return model, processor

    def _prepare_inputs(self, messages, processor, add_no_think_prompt: bool = False) -> dict:
        """Prepare model inputs from messages."""
        # Handle text-only vs multimodal
        if not self.router.has_visual_input(messages):
            messages = self.router._prepare_messages_for_text_model(messages)

        # Add system prompt to skip thinking (we already did latent reasoning)
        if add_no_think_prompt:
            no_think_system = {
                "role": "system",
                "content": "You have already deeply considered this question. Respond directly and concisely without any <think> tags or showing your reasoning process. Just give the answer."
            }
            messages = [no_think_system] + messages

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        return inputs

    def _extract_user_text(self, messages) -> str:
        """Extract text from the last user message for complexity analysis."""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', [])
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            return item.get('text', '')
        return ''

    def latent_thinking_step(
        self,
        model,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute one latent thinking step.

        Args:
            model: The Qwen model
            inputs_embeds: Current input embeddings [batch, seq, hidden]
            attention_mask: Attention mask [batch, seq]

        Returns:
            new_inputs_embeds: Extended embeddings with new hidden state
            new_attention_mask: Extended attention mask
        """
        with torch.no_grad():
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,  # Don't cache for thinking steps
                return_dict=True
            )

        # Get last hidden state from last layer, last position
        # Shape: [batch, 1, hidden_size]
        last_hidden = outputs.hidden_states[-1][:, -1:, :]

        # Ensure dtype matches
        if last_hidden.dtype != inputs_embeds.dtype:
            last_hidden = last_hidden.to(inputs_embeds.dtype)

        # Concatenate to inputs_embeds
        new_inputs_embeds = torch.cat([inputs_embeds, last_hidden], dim=1)

        # Extend attention mask
        batch_size = attention_mask.shape[0]
        new_attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=1)

        return new_inputs_embeds, new_attention_mask

    def latent_reasoning_generate(
        self,
        messages: List[dict],
        num_thinking_steps: Optional[int] = None,
        max_new_tokens: int = 2048,
        stream: bool = True,
        show_thinking_progress: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **generate_kwargs
    ) -> str:
        """
        Generate response with latent reasoning.

        Args:
            messages: Chat messages in Qwen format
            num_thinking_steps: Override adaptive thinking steps
            max_new_tokens: Maximum tokens for final response
            stream: Whether to stream output
            show_thinking_progress: Show "Thinking..." indicators
            temperature: Sampling temperature
            top_p: Top-p sampling
            **generate_kwargs: Additional args for model.generate()

        Returns:
            Generated text response
        """
        # Get model and processor
        model, processor = self._get_model_and_processor(messages)
        tokenizer = self.router.get_tokenizer(processor)

        # Determine thinking steps
        if num_thinking_steps is not None:
            thinking_steps = num_thinking_steps
        else:
            user_text = self._extract_user_text(messages)
            thinking_steps = self.estimate_complexity(user_text)

        # Prepare inputs
        inputs = self._prepare_inputs(messages, processor)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs.get(
            'attention_mask',
            torch.ones_like(input_ids)
        ).to(model.device)

        # Get input embeddings
        # Handle potential torch.compile wrapped models
        embed_layer = model.get_input_embeddings()
        if embed_layer is None:
            # Fallback for compiled models
            if hasattr(model, '_orig_mod'):
                embed_layer = model._orig_mod.get_input_embeddings()
            else:
                raise RuntimeError("Could not get input embeddings from model")

        with torch.no_grad():
            inputs_embeds = embed_layer(input_ids)

        # Ensure embeddings are in correct dtype
        if inputs_embeds.dtype != torch.bfloat16 and inputs_embeds.dtype != torch.float16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)

        # Show thinking progress
        if show_thinking_progress:
            print(f"\n[Thinking ({thinking_steps} steps)", end="", flush=True)

        start_time = time.time()

        # Latent reasoning loop
        for step in range(thinking_steps):
            inputs_embeds, attention_mask = self.latent_thinking_step(
                model=model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            if show_thinking_progress:
                print(".", end="", flush=True)

            if self.verbose:
                print(f"\n  Step {step + 1}: embeds shape {inputs_embeds.shape}")

        thinking_time = time.time() - start_time

        if show_thinking_progress:
            print(f"] ({thinking_time:.2f}s)\n", flush=True)

        # Append </think> embedding to signal thinking is complete
        # This tells the model to skip text-based thinking and answer directly
        think_complete_embed = self._get_think_complete_embedding(model, tokenizer)
        if think_complete_embed.dtype != inputs_embeds.dtype:
            think_complete_embed = think_complete_embed.to(inputs_embeds.dtype)

        inputs_embeds = torch.cat([inputs_embeds, think_complete_embed], dim=1)

        # Extend attention mask for the new tokens
        batch_size = attention_mask.shape[0]
        extra_len = think_complete_embed.shape[1]
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, extra_len), device=attention_mask.device, dtype=attention_mask.dtype)
        ], dim=1)

        # Generate text from enriched hidden states (model will answer directly)
        if stream:
            return self._generate_streaming(
                model, tokenizer, inputs_embeds, attention_mask,
                max_new_tokens, temperature, top_p, **generate_kwargs
            )
        else:
            return self._generate_text(
                model, tokenizer, inputs_embeds, attention_mask,
                max_new_tokens, temperature, top_p, **generate_kwargs
            )

    def _get_think_complete_embedding(self, model, tokenizer) -> torch.Tensor:
        """Get the embedding for </think> token to signal thinking is complete."""
        # Tokenize the end-of-think marker
        think_complete = "</think>\n\n"
        tokens = tokenizer.encode(think_complete, add_special_tokens=False, return_tensors="pt")
        tokens = tokens.to(model.device)

        # Get embeddings
        embed_layer = model.get_input_embeddings()
        if embed_layer is None and hasattr(model, '_orig_mod'):
            embed_layer = model._orig_mod.get_input_embeddings()

        with torch.no_grad():
            embeds = embed_layer(tokens)

        return embeds

    def _generate_streaming(
        self, model, tokenizer, inputs_embeds, attention_mask,
        max_new_tokens, temperature, top_p, **kwargs
    ) -> str:
        """Generate with streaming output, filtering residual think tags."""
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Run generation in background thread
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        thread = Thread(target=lambda: model.generate(**gen_kwargs))
        thread.start()

        # Stream with filtering
        full_response = ""
        started_content = False

        start_time = time.time()
        for text in streamer:
            # Filter out </think> and variations
            clean = re.sub(r'</?think>', '', text)

            # Skip leading whitespace until we get real content
            if not started_content:
                clean = clean.lstrip()
                if clean:
                    started_content = True

            if clean:
                print(clean, end='', flush=True)
                full_response += clean

        thread.join()
        end_time = time.time()

        if self.verbose:
            elapsed = end_time - start_time
            print(f"\n[Generated in {elapsed:.2f}s]")

        return full_response.strip()

    def _generate_text(
        self, model, tokenizer, inputs_embeds, attention_mask,
        max_new_tokens, temperature, top_p, **kwargs
    ) -> str:
        """Generate without streaming."""
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[:, inputs_embeds.shape[1]:]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return response


def create_latent_reasoner(
    precision: str = "bf16",
    min_steps: int = 3,
    max_steps: int = 100,  # Allow extensive thinking
    enable_tools: bool = False,
    mcp_client=None,
    verbose: bool = False,
    use_compile: bool = False  # Disabled by default for latent reasoning
) -> LatentReasoningQwen:
    """
    Create a latent reasoning instance.

    Args:
        precision: Model precision ("4bit", "fp8", "bf16")
        min_steps: Minimum thinking steps
        max_steps: Maximum thinking steps
        enable_tools: Enable tool calling
        mcp_client: MCP client instance
        verbose: Print detailed progress
        use_compile: Use torch.compile() (disabled by default, can cause issues)

    Returns:
        LatentReasoningQwen instance
    """
    from qwen_smart import QwenSmartRouter

    router = QwenSmartRouter(
        precision=precision,
        enable_tools=enable_tools,
        mcp_client=mcp_client,
        use_compile=use_compile  # Disable torch.compile for latent reasoning
    )
    return LatentReasoningQwen(
        router,
        min_steps=min_steps,
        max_steps=max_steps,
        verbose=verbose
    )


# Test/demo
if __name__ == "__main__":
    print("=" * 60)
    print("Latent Space Reasoning Demo")
    print("=" * 60)

    # Create reasoner with adaptive thinking
    print("\nInitializing latent reasoner...")
    reasoner = create_latent_reasoner(
        precision="bf16",
        min_steps=3,
        max_steps=15,
        verbose=True
    )

    # Test complexity estimation
    print("\n--- Complexity Estimation Tests ---")
    test_inputs = [
        "Hello!",
        "What is the capital of France?",
        "Explain how photosynthesis works",
        "What is 15 * 27?",
        "If all bloops are razzies, and all razzies are lazzies, are all bloops lazzies?",
        "Compare the pros and cons of nuclear energy versus renewable energy sources",
    ]

    for text in test_inputs:
        steps = reasoner.estimate_complexity(text)
        print(f"  '{text[:50]}...' -> {steps} steps")

    # Test actual generation
    print("\n--- Generation Test ---")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is 15 * 27? Think step by step."}
            ]
        }
    ]

    print("\nQuery: What is 15 * 27?")
    print("-" * 60)

    response = reasoner.latent_reasoning_generate(
        messages,
        max_new_tokens=256,
        stream=True
    )

    print("\n" + "=" * 60)
    print("Demo complete!")
