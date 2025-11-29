"""
MCP Client Wrapper
Manages connection to MCP search server and tool execution
"""

import asyncio
import json
import sys
import os
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPSearchClient:
    """
    Client wrapper for MCP search server
    Provides easy interface for chatbot to use web search
    """

    def __init__(self):
        """Initialize MCP client"""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools = []
        self._initialized = False

    async def initialize(self):
        """
        Initialize connection to MCP search server

        Returns:
            bool: True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Get the path to the search server
            tools_dir = os.path.dirname(os.path.abspath(__file__))
            server_script = os.path.join(tools_dir, "search_server.py")

            # Server parameters
            server_params = StdioServerParameters(
                command=sys.executable,  # Python interpreter
                args=[server_script],
                env=None
            )

            # Connect to server
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport

            # Create session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize session
            await self.session.initialize()

            # List available tools
            response = await self.session.list_tools()
            self.tools = response.tools

            print(f"\n[MCP] Connected to search server")
            print(f"[MCP] Available tools: {[t.name for t in self.tools]}\n")

            self._initialized = True
            return True

        except Exception as e:
            print(f"\n[MCP] Failed to initialize: {e}")
            print(f"[MCP] Web search will not be available\n")
            return False

    async def close(self):
        """Close MCP connection"""
        if self._initialized:
            await self.exit_stack.aclose()
            self._initialized = False

    def get_tools_description(self) -> str:
        """
        Get formatted description of available tools for system prompt

        Returns:
            Formatted tool descriptions
        """
        if not self.tools:
            return "No tools available."

        descriptions = []
        for tool in self.tools:
            desc = f"\nTool: {tool.name}\n"
            desc += f"Description: {tool.description}\n"
            desc += f"Parameters: {json.dumps(tool.inputSchema, indent=2)}\n"
            descriptions.append(desc)

        return "\n".join(descriptions)

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call a tool on the MCP server

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool

        Returns:
            Tool result as string
        """
        if not self._initialized:
            return "Error: MCP client not initialized"

        try:
            result = await self.session.call_tool(tool_name, arguments)

            # Extract text content from result
            if result and result.content:
                text_parts = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        text_parts.append(content.text)

                return "\n".join(text_parts)

            return "No result returned"

        except Exception as e:
            return f"Error calling tool '{tool_name}': {e}"

    async def search_web(self, query: str, max_results: int = 5, fetch_content: bool = True) -> str:
        """
        Convenience method for web search

        Args:
            query: Search query
            max_results: Maximum number of results
            fetch_content: Whether to fetch full page content

        Returns:
            Search results as formatted string
        """
        return await self.call_tool("web_search", {
            "query": query,
            "max_results": max_results,
            "fetch_content": fetch_content
        })

    async def fetch_url(self, url: str) -> str:
        """
        Convenience method for fetching a URL

        Args:
            url: URL to fetch

        Returns:
            Page content as string
        """
        return await self.call_tool("fetch_url", {
            "url": url
        })


class SyncMCPClient:
    """
    Synchronous wrapper around MCPSearchClient
    Allows using async MCP client in synchronous code
    """

    def __init__(self):
        """Initialize sync wrapper"""
        self.client = MCPSearchClient()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize MCP client

        Returns:
            bool: True if initialization successful
        """
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Initialize client
            result = self.loop.run_until_complete(self.client.initialize())
            self._initialized = result
            return result

        except Exception as e:
            print(f"[MCP] Sync initialization failed: {e}")
            return False

    def close(self):
        """Close MCP client"""
        if self._initialized and self.loop:
            self.loop.run_until_complete(self.client.close())
            self.loop.close()
            self._initialized = False

    def get_tools_description(self) -> str:
        """Get tool descriptions"""
        return self.client.get_tools_description()

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call a tool synchronously

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool result as string
        """
        if not self._initialized or not self.loop:
            return "Error: MCP client not initialized"

        try:
            return self.loop.run_until_complete(
                self.client.call_tool(tool_name, arguments)
            )
        except Exception as e:
            return f"Error: {e}"

    def search_web(self, query: str, max_results: int = 5, fetch_content: bool = True) -> str:
        """
        Search web synchronously

        Args:
            query: Search query
            max_results: Maximum results
            fetch_content: Whether to fetch content

        Returns:
            Search results
        """
        if not self._initialized or not self.loop:
            return "Error: MCP client not initialized"

        try:
            return self.loop.run_until_complete(
                self.client.search_web(query, max_results, fetch_content)
            )
        except Exception as e:
            return f"Error: {e}"

    def fetch_url(self, url: str) -> str:
        """
        Fetch URL synchronously

        Args:
            url: URL to fetch

        Returns:
            Page content
        """
        if not self._initialized or not self.loop:
            return "Error: MCP client not initialized"

        try:
            return self.loop.run_until_complete(
                self.client.fetch_url(url)
            )
        except Exception as e:
            return f"Error: {e}"


# Test function
async def test_client():
    """Test MCP client"""
    client = MCPSearchClient()

    try:
        # Initialize
        success = await client.initialize()
        if not success:
            print("Failed to initialize client")
            return

        # Test search
        print("\n=== Testing Web Search ===")
        result = await client.search_web("latest AI news", max_results=3)
        print(result)

        # Test URL fetch
        print("\n=== Testing URL Fetch ===")
        result = await client.fetch_url("https://www.example.com")
        print(result[:500])  # Print first 500 chars

    finally:
        await client.close()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_client())
