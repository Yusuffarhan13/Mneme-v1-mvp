"""
Terminal Chatbot using Qwen Smart Router
Supports text, image, and video inputs with 4-bit quantization
Includes autonomous web search powered by MCP
"""

from qwen_smart import QwenSmartRouter
import os
import sys
import argparse

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

try:
    from tools.mcp_client import SyncMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("[Warning] MCP not available. Web search will be disabled.")


class TerminalChatbot:
    def __init__(self, precision="4bit", enable_search=True):
        """
        Initialize chatbot

        Args:
            precision: Model precision ("4bit", "fp8", or "bf16")
            enable_search: Enable autonomous web search
        """
        # Initialize MCP client for web search
        self.mcp_client = None
        self.search_enabled = False

        if enable_search and MCP_AVAILABLE:
            try:
                print("\n[Initializing web search...]")
                self.mcp_client = SyncMCPClient()
                if self.mcp_client.initialize():
                    self.search_enabled = True
                    print("[Web search ready!]\n")
                else:
                    print("[Web search initialization failed]\n")
            except Exception as e:
                print(f"[Error initializing search: {e}]\n")

        # Initialize router with MCP client
        self.router = QwenSmartRouter(
            precision=precision,
            enable_tools=self.search_enabled,
            mcp_client=self.mcp_client
        )

        self.conversation_history = []
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "="*60)
        print("  QWEN TERMINAL CHATBOT (4-bit Quantization)")
        print("="*60)
        print("\nCommands:")
        print("  /exit or /quit - Exit the chatbot")
        print("  /clear - Clear conversation history")
        print("  /help - Show this help message")
        if self.search_enabled:
            print("  /search <query> - Manual web search")
        print("\nImage/Video Support:")
        print("  Just type the file path in your message:")
        print("  - Images: /path/to/image.jpg")
        print("  - Videos: /path/to/video.mp4")
        print("  - URLs: http://example.com/image.jpg")
        if self.search_enabled:
            print("\nWeb Search:")
            print("  Autonomous search enabled!")
            print("  I'll automatically search the web when needed.")
        print("\nTip: You can mix text with file paths!")
        print("="*60 + "\n")

    def detect_media_in_text(self, text):
        """Detect image/video file paths or URLs in text"""
        words = text.split()
        images = []
        videos = []
        clean_text_parts = []

        for word in words:
            # Check if it's a file path or URL
            is_media = False

            # Remove file:// prefix if present
            clean_word = word.replace('file://', '')

            # Check for image
            if any(clean_word.lower().endswith(ext) for ext in self.image_extensions):
                if clean_word.startswith('http://') or clean_word.startswith('https://'):
                    images.append(clean_word)
                    is_media = True
                elif os.path.exists(clean_word):
                    # Just use absolute path, no file:// prefix
                    images.append(os.path.abspath(clean_word))
                    is_media = True

            # Check for video
            elif any(clean_word.lower().endswith(ext) for ext in self.video_extensions):
                if clean_word.startswith('http://') or clean_word.startswith('https://'):
                    videos.append(clean_word)
                    is_media = True
                elif os.path.exists(clean_word):
                    # Just use absolute path, no file:// prefix
                    videos.append(os.path.abspath(clean_word))
                    is_media = True

            if not is_media:
                clean_text_parts.append(word)

        clean_text = ' '.join(clean_text_parts).strip()
        return clean_text, images, videos

    def build_message(self, user_input):
        """Build message in Qwen format"""
        clean_text, images, videos = self.detect_media_in_text(user_input)

        content = []

        # Add images
        for img in images:
            content.append({"type": "image", "image": img})

        # Add videos
        for vid in videos:
            content.append({"type": "video", "video": vid})

        # Add text (always include text, even if empty)
        if clean_text or not content:  # If no media, must have text
            content.append({"type": "text", "text": clean_text if clean_text else user_input})

        return {"role": "user", "content": content}

    def chat(self):
        """Main chat loop"""
        self.print_banner()

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['/exit', '/quit']:
                    print("\nGoodbye!")
                    # Cleanup MCP client
                    if self.mcp_client:
                        self.mcp_client.close()
                    break

                elif user_input.lower() == '/clear':
                    self.conversation_history = []
                    print("\n✓ Conversation history cleared!")
                    continue

                elif user_input.lower() == '/help':
                    self.print_banner()
                    continue

                elif user_input.startswith('/search ') and self.search_enabled:
                    # Manual search command
                    query = user_input[8:].strip()
                    if query:
                        print(f"\n[Searching: {query}]")
                        result = self.mcp_client.search_web(query, max_results=5)
                        print(f"\n{result}")

                        # Generate AI summary of the search results
                        print("\n" + "="*60)
                        print("AI Summary:")
                        print("="*60 + "\n")

                        summary_messages = [
                            {
                                "role": "user",
                                "content": [{
                                    "type": "text",
                                    "text": f"""Based on the following search results for "{query}", provide a clear and concise summary of what was found. Synthesize the key information, highlight the most important points, and cite the sources (URLs) where relevant.

Search Results:
{result}

Please provide a helpful summary that answers the original query."""
                                }]
                            }
                        ]

                        self.router.generate_response(
                            summary_messages,
                            max_new_tokens=1000,
                            show_prefix=False
                        )

                        print("\n" + "="*60)
                        print("Sources: See URLs in search results above")
                        print("="*60 + "\n")
                    else:
                        print("Usage: /search <query>")
                    continue

                # Build message
                message = self.build_message(user_input)

                # Add to conversation history
                self.conversation_history.append(message)

                # Generate response
                print("\nAssistant: ", end="", flush=True)

                if self.search_enabled:
                    # Use tool-enabled generation (autonomous search with unlimited iterations)
                    self.router.generate_with_tools(
                        self.conversation_history.copy(),
                        max_new_tokens=32000,  # Full 32k context window
                        max_iterations=999  # Effectively unlimited
                    )
                else:
                    # Regular generation without tools
                    self.router.generate_response(
                        self.conversation_history.copy(),
                        max_new_tokens=32000,  # Full 32k context window
                        show_prefix=False  # We already print "Assistant: "
                    )

            except KeyboardInterrupt:
                print("\n\nInterrupted! Type /exit to quit or continue chatting.")
                continue
            except Exception as e:
                print(f"\n\nError: {e}")
                print("Continuing chat...")
                continue

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen Terminal Chatbot with autonomous web search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chatbot.py                          # Standard mode with web search
  python chatbot.py --no-search              # Disable web search
  python chatbot.py -p 4bit                  # Use 4-bit quantization
        """
    )

    parser.add_argument(
        "--precision", "-p",
        choices=["4bit", "fp8", "bf16"],
        default="bf16",
        help="Model precision (default: bf16)"
    )

    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable web search"
    )

    return parser.parse_args()


def main():
    """Entry point with CLI argument support"""
    args = parse_args()

    chatbot = TerminalChatbot(
        precision=args.precision,
        enable_search=not args.no_search
    )
    chatbot.chat()


if __name__ == "__main__":
    main()
