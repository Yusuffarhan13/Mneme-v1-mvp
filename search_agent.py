"""
Fast Search Agent - Uses small 2B model for rapid web search
Handles autonomous web searching with minimal latency
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
import re
from datetime import datetime


class FastSearchAgent:
    """
    Lightweight search agent using 2B model for fast autonomous web search
    Optimized for speed and low latency
    """

    def __init__(self, precision="4bit", mcp_client=None):
        """
        Initialize fast search agent with small model

        Args:
            precision: "4bit" (recommended), "fp8", or "bf16"
            mcp_client: MCP client for web search
        """
        self.precision = precision
        self.mcp_client = mcp_client
        self.model = None
        self.tokenizer = None
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Fast 1.5B parameter model
        print(f"[Search Agent] Initializing with {self.model_name}")

    def load_model(self):
        """Load the small fast model"""
        if self.model is None:
            print(f"[Search Agent] Loading {self.model_name} with {self.precision} precision...")

            if self.precision == "4bit":
                # 4-bit quantization - only ~1GB VRAM!
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"[Search Agent] Model loaded (~1GB VRAM for 4-bit)")

        return self.model, self.tokenizer

    def create_search_prompt(self, query: str, search_results: list = None) -> str:
        """
        Create optimized prompt for search agent

        Args:
            query: User's query
            search_results: Previous search results (if any)

        Returns:
            Formatted prompt
        """
        current_date = datetime.now().strftime("%B %d, %Y")

        if search_results:
            # Synthesis mode
            prompt = f"""You are analyzing web search results. Today is {current_date}.

User Query: {query}

Search Results:
{search_results[:2000]}  # Limit context

Task: Provide a concise, accurate answer based on these results. Cite sources."""
        else:
            # Search planning mode
            prompt = f"""You are a web search assistant. Today is {current_date}.

User Query: {query}

Task: Output ONLY a JSON search query in this format:
{{"search": "optimized search query here"}}

Make the search query specific and effective. Output only valid JSON, nothing else."""

        return prompt

    def should_search_more(self, query: str, gathered_info: str, iteration: int) -> dict:
        """
        Decide if more searches are needed

        Args:
            query: Original query
            gathered_info: Information gathered so far
            iteration: Current iteration number

        Returns:
            dict with 'need_more' (bool) and 'reason' (str)
        """
        # Simple heuristic for now
        if iteration >= 5:  # Safety limit
            return {"need_more": False, "reason": "Max searches reached"}

        if not gathered_info or len(gathered_info) < 500:
            return {"need_more": True, "reason": "Need more information"}

        # Check if query is answered
        query_terms = query.lower().split()
        info_lower = gathered_info.lower()
        coverage = sum(1 for term in query_terms if term in info_lower)

        if coverage < len(query_terms) * 0.7:
            return {"need_more": True, "reason": "Query not fully covered"}

        return {"need_more": False, "reason": "Sufficient information gathered"}

    def generate_search_query(self, user_query: str, iteration: int = 0) -> str:
        """
        Generate optimized search query using fast model

        Args:
            user_query: User's original query
            iteration: Current search iteration

        Returns:
            Optimized search query
        """
        model, tokenizer = self.load_model()

        # For first iteration, use user query directly for speed
        if iteration == 0:
            return user_query

        # For subsequent iterations, slightly modify query
        variations = [
            f"{user_query} latest updates",
            f"{user_query} detailed information",
            f"{user_query} examples",
            f"{user_query} explained",
            f"{user_query} best practices"
        ]

        return variations[min(iteration - 1, len(variations) - 1)]

    def extract_search_query(self, response: str) -> str:
        """
        Extract search query from model response

        Args:
            response: Model's response

        Returns:
            Extracted search query
        """
        # Try to parse JSON
        try:
            match = re.search(r'\{.*?"search":\s*"([^"]+)".*?\}', response, re.DOTALL)
            if match:
                return match.group(1)
        except Exception:
            pass

        # Fallback: return cleaned response
        return response.strip()

    def autonomous_search(self, user_query: str, max_searches: int = 10) -> tuple:
        """
        Perform autonomous web search with unlimited iterations

        Args:
            user_query: User's query
            max_searches: Maximum number of searches (safety limit)

        Returns:
            (all_results: str, search_count: int)
        """
        print(f"\n[Search Agent] Starting autonomous search for: '{user_query}'")

        all_results = []
        search_count = 0

        for iteration in range(max_searches):
            # Generate search query
            search_query = self.generate_search_query(user_query, iteration)

            print(f"\n[Search Agent] Iteration {iteration + 1}: Searching '{search_query}'")

            # Execute search
            if self.mcp_client:
                try:
                    result = self.mcp_client.search_web(search_query, max_results=5)
                    all_results.append(result)
                    search_count += 1
                    print(f"[Search Agent] ✓ Search {search_count} complete")
                except Exception as e:
                    print(f"[Search Agent] ✗ Search failed: {e}")
                    break

            # Decide if we need more searches
            gathered_info = "\n\n".join(all_results)
            decision = self.should_search_more(user_query, gathered_info, iteration + 1)

            if not decision["need_more"]:
                print(f"[Search Agent] Search complete: {decision['reason']}")
                break

            print(f"[Search Agent] Continuing: {decision['reason']}")

        # Combine all results
        combined_results = "\n\n" + "="*80 + "\n\n".join(all_results)

        print(f"\n[Search Agent] ✓ Completed {search_count} searches")
        print(f"[Search Agent] Gathered {len(combined_results)} characters of information")

        return combined_results, search_count


# Test function
def test_search_agent():
    """Test the fast search agent"""
    from tools.mcp_client import SyncMCPClient

    # Initialize MCP
    client = SyncMCPClient()
    if not client.initialize():
        print("Failed to initialize MCP")
        return

    # Initialize search agent
    agent = FastSearchAgent(precision="4bit", mcp_client=client)

    # Test autonomous search
    results, count = agent.autonomous_search("Python asyncio best practices", max_searches=3)

    print("\n" + "="*80)
    print(f"RESULTS ({count} searches):")
    print("="*80)
    print(results[:1000])  # Print first 1000 chars

    # Cleanup
    client.close()


if __name__ == "__main__":
    test_search_agent()
