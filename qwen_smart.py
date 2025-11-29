"""
Qwen Smart Router - Automatic Model Selection with Quantization Support

Precision Options:
- "4bit": Ultra-efficient 4-bit quantization (8B: ~3-4GB, 4B: ~2-3GB VRAM)
- "fp8":  Official FP8 models (8B: ~4-5GB, 4B: ~3-4GB VRAM)
- "bf16": Full precision bfloat16 (8B: ~10-12GB, 4B: ~5-6GB VRAM)

Automatic Routing:
- Text-only inputs → Qwen3-8B (text model, better reasoning)
- Image/Video inputs → Qwen3-4B-VL-Thinking (vision-language model)

Requirements:
- transformers >= 4.57.0
- bitsandbytes (for 4-bit quantization)
- accelerate
"""

from transformers import Qwen3VLForConditionalGeneration, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TextStreamer, BitsAndBytesConfig
import torch
import json
import re
from datetime import datetime
import time
import warnings

# Enable CUDA optimizations for RTX 4090
if torch.cuda.is_available():
    # Enable cuDNN auto-tuner for optimal CUDA kernels
    torch.backends.cudnn.benchmark = True
    # Enable TensorFloat-32 for faster matrix operations on RTX 30/40 series
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"CUDA Optimizations: ✓ cuDNN benchmark, ✓ TF32 on RTX 4090")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Check for Flash Attention support
# PyTorch 2.0+ has built-in Flash Attention via SDPA (Scaled Dot Product Attention)
FLASH_ATTENTION_AVAILABLE = False
SDPA_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
    print("Flash Attention 2: ✓ Available (expect 2-4x speedup)")
except ImportError:
    # Check for PyTorch's built-in SDPA (almost as good as flash-attn)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        SDPA_AVAILABLE = True
        print("Flash Attention (SDPA): ✓ Using PyTorch built-in (expect 1.5-3x speedup)")
        print("  Note: For best performance, ensure PyTorch >= 2.0 with CUDA support")
    else:
        print("Flash Attention: ✗ Not available")
        print("  Tip: Upgrade to PyTorch 2.0+ for built-in Flash Attention support")

class QwenSmartRouter:
    def __init__(self, precision="4bit", enable_tools=False, mcp_client=None, use_compile=True):
        """
        Initialize router with precision setting
        Args:
            precision: "bf16" (full), "fp8" (official), or "4bit" (most efficient)
            enable_tools: Enable tool calling support
            mcp_client: MCP client instance for web search
            use_compile: Use torch.compile() for 1.5-2x speedup (requires PyTorch 2.0+)
        """
        self.precision = precision
        self.model_8b = None
        self.processor_8b = None
        self.model_4b = None
        self.processor_4b = None
        self.enable_tools = enable_tools
        self.mcp_client = mcp_client
        self.use_compile = use_compile

        print(f"Initialized with precision: {precision}")
        if enable_tools:
            print("Tool calling enabled - model decides when to search")
        if use_compile:
            print("torch.compile(): ✓ Enabled (PyTorch 2.0+ optimization)")

    def load_8b_model(self):
        """Load Qwen3-8B for text and image inputs"""
        if self.model_8b is None:
            if self.precision == "4bit":
                print("Loading Qwen3-4B model with 4-bit quantization (32k context)...")
                # Configure 4-bit quantization with NF4
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
                )

                # Add Flash Attention if available (flash-attn or PyTorch SDPA)
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                }
                if FLASH_ATTENTION_AVAILABLE:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                elif SDPA_AVAILABLE:
                    model_kwargs["attn_implementation"] = "sdpa"

                self.model_8b = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3-4B",
                    **model_kwargs
                )

                # Apply torch.compile() for additional speedup
                if self.use_compile and hasattr(torch, 'compile'):
                    print("Compiling model with torch.compile() (first run will be slower)...")
                    self.model_8b = torch.compile(self.model_8b, mode="reduce-overhead")

                self.processor_8b = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
                print("4B model loaded with 4-bit quantization (~3-4GB VRAM, 32k context)")

            elif self.precision == "fp8":
                print("Loading Qwen3-4B-FP8 model (32k context)...")

                model_kwargs = {
                    "torch_dtype": torch.float8_e4m3fn,
                    "device_map": "cuda",
                    "low_cpu_mem_usage": True,
                }
                if FLASH_ATTENTION_AVAILABLE:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                elif SDPA_AVAILABLE:
                    model_kwargs["attn_implementation"] = "sdpa"

                self.model_8b = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3-4B-FP8",
                    **model_kwargs
                )

                if self.use_compile and hasattr(torch, 'compile'):
                    print("Compiling model with torch.compile()...")
                    self.model_8b = torch.compile(self.model_8b, mode="reduce-overhead")

                self.processor_8b = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-FP8")
                print("4B FP8 model loaded (~4-5GB VRAM, 32k context)")

            else:  # bf16
                print("Loading Qwen3-4B model (bf16, 32k context)...")

                model_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "cuda",
                    "low_cpu_mem_usage": True,
                }
                if FLASH_ATTENTION_AVAILABLE:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                elif SDPA_AVAILABLE:
                    model_kwargs["attn_implementation"] = "sdpa"

                self.model_8b = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3-4B",
                    **model_kwargs
                )

                if self.use_compile and hasattr(torch, 'compile'):
                    print("Compiling model with torch.compile()...")
                    self.model_8b = torch.compile(self.model_8b, mode="reduce-overhead")

                self.processor_8b = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
                print("4B model loaded with bf16 (~10-12GB VRAM, 32k context)")

        return self.model_8b, self.processor_8b

    def load_4b_model(self):
        """Load Qwen3-4B-VL-Thinking for video inputs"""
        if self.model_4b is None:
            if self.precision == "4bit":
                print("Loading Qwen3-4B-VL-Thinking model with 4-bit quantization (32k context)...")
                # Configure 4-bit quantization with NF4
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
                )

                # Add Flash Attention if available (flash-attn or PyTorch SDPA)
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                }
                if FLASH_ATTENTION_AVAILABLE:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                elif SDPA_AVAILABLE:
                    model_kwargs["attn_implementation"] = "sdpa"

                self.model_4b = Qwen3VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen3-VL-4B-Thinking",
                    **model_kwargs
                )

                # Apply torch.compile() for additional speedup
                if self.use_compile and hasattr(torch, 'compile'):
                    print("Compiling VL model with torch.compile()...")
                    self.model_4b = torch.compile(self.model_4b, mode="reduce-overhead")

                self.processor_4b = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Thinking")
                print("4B VL model loaded with 4-bit quantization (~2-3GB VRAM, 32k context)")

            elif self.precision == "fp8":
                print("Loading Qwen3-4B-VL-Thinking-FP8 model (32k context)...")

                model_kwargs = {
                    "torch_dtype": torch.float8_e4m3fn,
                    "device_map": "cuda",
                    "low_cpu_mem_usage": True,
                }
                if FLASH_ATTENTION_AVAILABLE:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                elif SDPA_AVAILABLE:
                    model_kwargs["attn_implementation"] = "sdpa"

                self.model_4b = Qwen3VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen3-VL-4B-Thinking-FP8",
                    **model_kwargs
                )

                if self.use_compile and hasattr(torch, 'compile'):
                    print("Compiling VL model with torch.compile()...")
                    self.model_4b = torch.compile(self.model_4b, mode="reduce-overhead")

                self.processor_4b = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Thinking-FP8")
                print("4B VL FP8 model loaded (~3-4GB VRAM, 32k context)")

            else:  # bf16
                print("Loading Qwen3-4B-VL-Thinking model (bf16, 32k context)...")

                model_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "cuda",
                    "low_cpu_mem_usage": True,
                }
                if FLASH_ATTENTION_AVAILABLE:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                elif SDPA_AVAILABLE:
                    model_kwargs["attn_implementation"] = "sdpa"

                self.model_4b = Qwen3VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen3-VL-4B-Thinking",
                    **model_kwargs
                )

                if self.use_compile and hasattr(torch, 'compile'):
                    print("Compiling VL model with torch.compile()...")
                    self.model_4b = torch.compile(self.model_4b, mode="reduce-overhead")

                self.processor_4b = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Thinking")
                print("4B VL model loaded with bf16 (~5-6GB VRAM, 32k context)")

        return self.model_4b, self.processor_4b

    def has_video(self, messages):
        """Check if messages contain video input"""
        for message in messages:
            if "content" in message:
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "video":
                        return True
        return False

    def has_visual_input(self, messages):
        """Check if messages contain image or video input"""
        for message in messages:
            if "content" in message:
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") in ["image", "video"]:
                        return True
        return False

    def get_tokenizer(self, processor):
        """Get tokenizer from processor or tokenizer object"""
        # For text-only models, processor IS the tokenizer
        # For vision models, processor.tokenizer is the tokenizer
        if hasattr(processor, 'tokenizer'):
            return processor.tokenizer
        return processor

    def get_system_prompt_with_tools(self):
        """
        Create system prompt with tool definitions for autonomous tool calling

        Returns:
            System prompt string
        """
        if not self.enable_tools or not self.mcp_client:
            return ""

        tools_desc = self.mcp_client.get_tools_description()

        # Get current date for temporal awareness
        current_date = datetime.now().strftime("%B %d, %Y")

        system_prompt = f"""You are a helpful assistant with web search capabilities. Today's date is {current_date}.

IMPORTANT: When searching for current/recent news, use RELATIVE time terms like:
- "latest news" or "breaking news" (NOT "news for November 3 2025")
- "current events" or "news today"
- "recent developments"
Never include specific dates in news queries unless the user explicitly asks for a specific date.

CITATION REQUIREMENT:
• ALWAYS cite URLs in your response when using search results
• Format: "According to [URL], ..." or "Source: [URL]"
• Show what information came from which URL
• Users want to verify sources

SIMPLE RULES:
• If you know the answer → Just answer it (keep it brief)
• If you need current info → Use web_search tool with relative time terms
• Keep thinking minimal
• After searching, cite your sources with URLs

SEARCH FOR:
- Current events, news (use "latest news", "breaking news today")
- Real-time data (prices, weather, scores)
- Recent developments (use "recent" not specific dates)

DON'T SEARCH FOR:
- General knowledge (history, science, programming)
- Math, definitions, established concepts

Available Tools:
{tools_desc}

TOOL CALL FORMAT:
When you need to call a tool, use this EXACT format:
<tool_call>{{"name": "tool_name", "arguments": {{"param": "value"}}}}</tool_call>

Example:
<tool_call>{{"name": "web_search", "arguments": {{"query": "latest breaking news", "max_results": 5}}}}</tool_call>

REMEMBER: Always cite URLs when providing search-based information!"""
        return system_prompt

    def extract_tool_call_for_parsing(self, response_text):
        """
        Extract just the tool call for parsing, but keep thinking tags in original response

        Args:
            response_text: Raw response from model

        Returns:
            Text with tool call extracted for parsing
        """
        # Keep original response intact, just extract tool call section
        # We don't remove thinking tags - user wants to see them
        return response_text.strip()

    def parse_tool_call(self, response_text):
        """
        Parse tool call from model response (without modifying the original text)

        Args:
            response_text: Generated text from model

        Returns:
            dict with 'name' and 'arguments', or None if no tool call
        """
        # Don't clean - user wants to see thinking blocks
        # Just extract the tool call

        # Look for <tool_call>...</tool_call> pattern
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, response_text, re.DOTALL)

        if not matches:
            return None

        try:
            # Parse JSON from first match, stripping whitespace
            json_str = matches[0].strip()
            tool_call = json.loads(json_str)
            if 'name' in tool_call and 'arguments' in tool_call:
                return tool_call
            else:
                print(f"\n[Warning: Invalid tool call format - missing name or arguments]")
                return None
        except json.JSONDecodeError as e:
            print(f"\n[Warning: Could not parse tool call JSON: {e}]")
            print(f"[Debug: Attempted to parse: {json_str[:100]}...]")
            return None

    def _prepare_messages_for_text_model(self, messages):
        """Convert list-based content to string for text-only model"""
        clean_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if isinstance(msg.get('content'), list):
                # Extract text parts
                text_parts = [item['text'] for item in msg['content'] if item.get('type') == 'text']
                new_msg['content'] = "\n".join(text_parts)
            clean_messages.append(new_msg)
        return clean_messages

    def generate_text_only(self, messages, max_new_tokens=32000):
        """
        Generate response and return text (no streaming)

        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text string
        """
        # Determine which model to use
        if self.has_visual_input(messages):
            model, processor = self.load_4b_model()
            final_messages = messages
        else:
            model, processor = self.load_8b_model()
            final_messages = self._prepare_messages_for_text_model(messages)

        # Prepare inputs
        inputs = processor.apply_chat_template(
            final_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate without streaming - optimized for speed
        # Use lower temperature for more deterministic, concise outputs
        tokenizer = self.get_tokenizer(processor)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,  # Very low temperature to reduce hallucination
                top_p=0.8,
                repetition_penalty=1.2,  # Higher penalty to prevent thinking loops
                use_cache=True,  # Enable KV cache for faster generation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        response_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        return response_text

    def generate_streaming(self, messages, max_new_tokens=32000, show_perf=False):
        """
        Generate response with live streaming output

        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum tokens to generate
            show_perf: Show performance metrics (tokens/sec)
        """
        # Determine which model to use
        if self.has_visual_input(messages):
            model, processor = self.load_4b_model()
            final_messages = messages
        else:
            model, processor = self.load_8b_model()
            final_messages = self._prepare_messages_for_text_model(messages)

        # Prepare inputs
        inputs = processor.apply_chat_template(
            final_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # Initialize streamer for live output
        tokenizer = self.get_tokenizer(processor)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate with live streaming - optimized parameters
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,  # Enable KV cache for faster generation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        end_time = time.time()

        # Performance metrics
        if show_perf:
            num_tokens = generated_ids.shape[1] - inputs['input_ids'].shape[1]
            elapsed = end_time - start_time
            tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
            print(f"\n[Performance: {tokens_per_sec:.1f} tokens/sec, {num_tokens} tokens in {elapsed:.2f}s]")

        print()  # New line after generation

    def generate_with_tools(self, messages, max_new_tokens=32000, max_iterations=999):
        """
        Generate response with autonomous tool calling (ReAct pattern)
        The model decides when to search using decide_search tool
        Unlimited iterations!

        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum tokens per generation
            max_iterations: Maximum number of tool calling iterations (default: unlimited)

        Returns:
            Final response text
        """
        if not self.enable_tools or not self.mcp_client:
            # Fall back to regular generation
            return self.generate_response(messages, max_new_tokens, show_prefix=True)

        # Add system prompt with tools
        system_prompt = self.get_system_prompt_with_tools()
        messages_with_system = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            }
        ] + messages.copy()

        for iteration in range(max_iterations):
            # Generate response (non-streaming to check for tool calls)
            # Use shorter max_tokens to prevent excessive thinking
            try:
                response_text = self.generate_text_only(messages_with_system, max_new_tokens=2000)
            except Exception as e:
                print(f"\nError generating response: {e}")
                return "Generation error"

            # Check for tool calls
            tool_call = self.parse_tool_call(response_text)

            if tool_call is None:
                # No tool call - this is the final answer
                # Regenerate with streaming for live typing effect
                try:
                    self.generate_streaming(messages_with_system, max_new_tokens)
                except Exception as e:
                    print(f"\nError streaming response: {e}")
                    # Fallback to non-streaming
                    print(response_text)
                return "Generation complete"

            # Tool call detected - execute it
            tool_name = tool_call['name']
            tool_args = tool_call['arguments']

            # Display search activity
            if tool_name == "web_search":
                query = tool_args.get('query', 'unknown')
                print(f"\n[Searching: {query}]", flush=True)
            else:
                print(f"\n[Calling tool: {tool_name}]", flush=True)

            # Call tool via MCP client
            try:
                tool_result = self.mcp_client.call_tool(tool_name, tool_args)
                print(f"[Search complete - found results]\n", flush=True)
            except Exception as e:
                print(f"\n[Search failed: {e}]")
                tool_result = f"Error: Could not complete search - {str(e)}"

            # Add assistant's tool call to conversation
            messages_with_system.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}]
            })

            # Add tool result as user message
            messages_with_system.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"Tool Result:\n{tool_result}\n\nNow answer the original question clearly and concisely."
                }]
            })

        # Max iterations reached
        print("I apologize, but I've reached the maximum number of tool calls. Please try rephrasing your question.")
        return "Max iterations reached"

    def generate_response(self, messages, max_new_tokens=512, show_prefix=True, show_perf=False):
        """
        Generate response with automatic model routing
        - Images/Videos -> Qwen3-4B-VL-Thinking
        - Text-only -> Qwen3-8B

        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum tokens to generate
            show_prefix: Whether to print "Response: " prefix
            show_perf: Show performance metrics (tokens/sec)
        """
        # Determine which model to use
        if self.has_visual_input(messages):
            model, processor = self.load_4b_model()
            final_messages = messages
        else:
            model, processor = self.load_8b_model()
            final_messages = self._prepare_messages_for_text_model(messages)

        # Prepare inputs
        inputs = processor.apply_chat_template(
            final_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # Initialize streamer for live output
        tokenizer = self.get_tokenizer(processor)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate with live streaming - optimized parameters
        if show_prefix:
            print("\nResponse: ", end="", flush=True)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,  # Enable KV cache
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        end_time = time.time()

        # Performance metrics
        if show_perf:
            num_tokens = generated_ids.shape[1] - inputs['input_ids'].shape[1]
            elapsed = end_time - start_time
            tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
            print(f"\n[Performance: {tokens_per_sec:.1f} tokens/sec, {num_tokens} tokens in {elapsed:.2f}s]")

        print()  # New line after generation completes

        return "Generation complete!"


# Example usage
if __name__ == "__main__":
    # Choose precision: "4bit" (most efficient), "fp8" (balanced), or "bf16" (full quality)
    router = QwenSmartRouter(precision="4bit")  # Uses ~2-4GB VRAM total!

    # Example 1: Image input (uses 8B model)
    print("\n=== Example 1: Image Input ===")
    messages_image = [
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
    router.generate_response(messages_image)

    # Example 2: Video input (uses 4B model)
    print("\n=== Example 2: Video Input ===")
    messages_video = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
                },
                {"type": "text", "text": "Describe what happens in this video."},
            ],
        }
    ]
    router.generate_response(messages_video)

    # Example 3: Text-only input (uses 8B model)
    print("\n=== Example 3: Text-Only Input ===")
    messages_text = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the capital of France?"},
            ],
        }
    ]
    router.generate_response(messages_text)
