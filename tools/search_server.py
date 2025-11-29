"""
MCP Search Server - Enhanced Multi-Source Edition
Implements Model Context Protocol server for web search and content fetching
Searches across 20+ sources including Reddit, LinkedIn, GitHub, Stack Overflow, etc.
"""

import asyncio
import json
from typing import Any, Sequence
from concurrent.futures import ThreadPoolExecutor
import httpx
from ddgs import DDGS
from functools import lru_cache
import hashlib
import time

from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from mcp.server.stdio import stdio_server

try:
    from .content_processor import ContentProcessor
except ImportError:
    from content_processor import ContentProcessor


class SearchServer:
    """MCP Server for multi-source web search functionality"""

    # Major websites to search across
    # Fast mode: Top 10 most valuable sources for speed
    SEARCH_SOURCES_FAST = [
        "reddit.com",
        "stackoverflow.com",
        "github.com",
        "wikipedia.org",
        "medium.com",
        "linkedin.com",
        "arxiv.org",
        "techcrunch.com",
        "youtube.com",
        "dev.to",
    ]

    # Full mode: All 20 sources for comprehensive results
    SEARCH_SOURCES_FULL = [
        "reddit.com",
        "linkedin.com",
        "github.com",
        "stackoverflow.com",
        "stackexchange.com",
        "medium.com",
        "dev.to",
        "hackernews.ycombinator.com",
        "quora.com",
        "wikipedia.org",
        "youtube.com",
        "twitter.com",
        "facebook.com",
        "instagram.com",
        "arxiv.org",
        "nature.com",
        "sciencedirect.com",
        "forbes.com",
        "techcrunch.com",
        "wired.com",
    ]

    # Universal mode: Search ALL websites (no restrictions)
    # Uses only general DuckDuckGo search for maximum coverage and speed
    SEARCH_SOURCES = []  # Empty = search all websites, not limited to specific sites

    def __init__(self, max_results: int = 5, fetch_timeout: int = 10):
        """
        Initialize enhanced search server with multi-source capability and caching

        Args:
            max_results: Maximum number of search results per source
            fetch_timeout: Timeout for fetching page content (seconds)
        """
        self.max_results = max_results
        self.fetch_timeout = fetch_timeout
        self.content_processor = ContentProcessor(max_content_length=1500)  # Reduced for faster, concise results
        self.executor = ThreadPoolExecutor(max_workers=20)  # Increased for parallel searches
        self.server = Server("search-server")

        # Search cache: query_hash -> (results, timestamp)
        self.search_cache = {}
        self.cache_ttl = 300  # 5 minutes cache

        # Register handlers
        self.setup_handlers()

    def setup_handlers(self):
        """Setup MCP request handlers"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="web_search",
                    description=(
                        "Search the web for current information, facts, news, or any data "
                        "not in your training data. Automatically fetches and parses content "
                        "from top results. Use this when you need up-to-date information or "
                        "when the user asks about recent events, specific facts, or current topics."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 5)",
                                "default": 5
                            },
                            "fetch_content": {
                                "type": "boolean",
                                "description": "Whether to fetch full page content (default: true)",
                                "default": True
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="fetch_url",
                    description=(
                        "Fetch and extract content from a specific URL. "
                        "Use this to get detailed information from a known web page."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to fetch"
                            }
                        },
                        "required": ["url"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool calls"""

            if name == "web_search":
                query = arguments.get("query", "")
                max_results = arguments.get("max_results", self.max_results)
                fetch_content = arguments.get("fetch_content", True)

                if not query:
                    return [TextContent(type="text", text="Error: Query cannot be empty")]

                # Perform search
                results = await self.search_web(query, max_results, fetch_content)

                if not results:
                    return [TextContent(
                        type="text",
                        text=f"No results found for query: '{query}'"
                    )]

                # Format results
                formatted = self.content_processor.format_search_results(results, query)

                return [TextContent(type="text", text=formatted)]

            elif name == "fetch_url":
                url = arguments.get("url", "")

                if not url:
                    return [TextContent(type="text", text="Error: URL cannot be empty")]

                # Fetch content
                content = await self.fetch_page_content(url)

                if content:
                    return [TextContent(type="text", text=f"Content from {url}:\n\n{content}")]
                else:
                    return [TextContent(type="text", text=f"Error: Could not fetch content from {url}")]

            else:
                return [TextContent(type="text", text=f"Error: Unknown tool '{name}'")]

    def _get_cache_key(self, query: str, max_results: int, fetch_content: bool) -> str:
        """Generate cache key for query"""
        key_str = f"{query}:{max_results}:{fetch_content}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_results(self, cache_key: str) -> list[dict] | None:
        """Get cached results if available and not expired"""
        if cache_key in self.search_cache:
            results, timestamp = self.search_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return results
            else:
                # Expired, remove from cache
                del self.search_cache[cache_key]
        return None

    def _cache_results(self, cache_key: str, results: list[dict]):
        """Cache search results"""
        self.search_cache[cache_key] = (results, time.time())

    async def search_web(self, query: str, max_results: int = 5, fetch_content: bool = True) -> list[dict]:
        """
        Enhanced multi-source web search across 20+ websites in parallel with caching

        Args:
            query: Search query
            max_results: Maximum number of results per source
            fetch_content: Whether to fetch full page content

        Returns:
            Aggregated and ranked search results from multiple sources
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(query, max_results, fetch_content)
            cached_results = self._get_cached_results(cache_key)
            if cached_results:
                import sys
                print(f"[Cache Hit] Returning cached results for '{query}'", file=sys.stderr, flush=True)
                return cached_results
            # Launch parallel searches across all sources
            search_tasks = []

            if not self.SEARCH_SOURCES:
                # Universal mode: Search all websites with higher limit
                # Get more results from general search for better coverage
                search_tasks.append(self._search_general(query, max_results * 3))
                import sys
                print(f"[Universal Search] Searching all websites (no restrictions)...", file=sys.stderr, flush=True)
            else:
                # Multi-source mode: General + specific sites
                search_tasks.append(self._search_general(query, max_results))

                # Site-specific searches for configured sources
                for source in self.SEARCH_SOURCES:
                    search_tasks.append(self._search_site_specific(query, source, 2))  # 2 results per site

                import sys
                print(f"[Multi-Source Search] Querying {len(search_tasks)} sources in parallel...", file=sys.stderr, flush=True)

            # Execute all searches in parallel
            all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Aggregate results from all sources
            aggregated_results = []
            for results in all_results:
                if isinstance(results, Exception):
                    continue
                if results:
                    aggregated_results.extend(results)

            if not aggregated_results:
                return []

            # Remove duplicates based on URL
            seen_urls = set()
            unique_results = []
            for result in aggregated_results:
                url = result.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)

            # Rank results (basic relevance scoring)
            ranked_results = self._rank_results(unique_results, query)

            # Limit to total max results
            top_results = ranked_results[:max_results * 3]  # Get more results for variety

            print(f"[Multi-Source Search] Found {len(unique_results)} unique results from {len(aggregated_results)} total", file=sys.stderr, flush=True)

            # If fetch_content is False, return ranked results
            if not fetch_content:
                return top_results[:max_results]

            # Fetch full content from top results in parallel
            urls = [r['url'] for r in top_results[:max_results]]
            contents = await self.fetch_multiple_urls(urls)

            # Combine results with fetched content
            enhanced_results = []
            for result, content in zip(top_results[:max_results], contents):
                enhanced_result = {
                    'title': result['title'],
                    'url': result['url'],
                    'snippet': result['snippet'],
                    'content': content or result['snippet'],  # Fallback to snippet
                    'source': result.get('source', 'unknown')
                }
                enhanced_results.append(enhanced_result)

            # Cache the results
            self._cache_results(cache_key, enhanced_results)

            return enhanced_results

        except Exception as e:
            print(f"Error in multi-source web search: {e}")
            return []

    async def _search_general(self, query: str, max_results: int) -> list[dict]:
        """
        General DuckDuckGo search

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._ddg_search,
                query,
                max_results
            )
            # Add source information
            for result in results:
                result['source'] = 'DuckDuckGo'
            return results
        except Exception as e:
            print(f"Error in general search: {e}")
            return []

    async def _search_site_specific(self, query: str, site: str, max_results: int = 2) -> list[dict]:
        """
        Search within a specific website using DuckDuckGo site: operator

        Args:
            query: Search query
            site: Website domain (e.g., 'reddit.com')
            max_results: Maximum results from this site

        Returns:
            List of search results from specific site
        """
        try:
            site_query = f"{query} site:{site}"
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._ddg_search,
                site_query,
                max_results
            )
            # Add source information
            for result in results:
                result['source'] = site
            return results
        except Exception as e:
            # Fail silently for individual site searches
            return []

    def _rank_results(self, results: list[dict], query: str) -> list[dict]:
        """
        Rank results by relevance using simple scoring

        Args:
            results: List of search results
            query: Original search query

        Returns:
            Sorted list of results by relevance score
        """
        query_terms = query.lower().split()

        for result in results:
            score = 0
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            url = result.get('url', '').lower()

            # Score based on query term matches
            for term in query_terms:
                if term in title:
                    score += 10  # Title matches are most important
                if term in snippet:
                    score += 3   # Snippet matches
                if term in url:
                    score += 2   # URL matches

            # Boost certain high-quality sources
            source = result.get('source', '')
            if source in ['stackoverflow.com', 'github.com', 'wikipedia.org']:
                score += 5
            elif source in ['reddit.com', 'linkedin.com', 'medium.com']:
                score += 3

            result['relevance_score'] = score

        # Sort by relevance score (descending)
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)

    def _ddg_search(self, query: str, max_results: int) -> list[dict]:
        """
        Synchronous DuckDuckGo search (runs in thread pool)

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': r.get('title', 'Untitled'),
                        'url': r.get('href', ''),
                        'snippet': r.get('body', '')
                    })
            return results
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []

    async def fetch_page_content(self, url: str) -> str | None:
        """
        Fetch and extract content from a URL

        Args:
            url: URL to fetch

        Returns:
            Extracted content or None if failed
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.fetch_timeout,
                follow_redirects=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Extract content
                html = response.text
                content = self.content_processor.extract_main_content(html, url)

                return content

        except httpx.TimeoutException:
            print(f"Timeout fetching {url}")
            return None
        except httpx.HTTPError as e:
            print(f"HTTP error fetching {url}: {e}")
            return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    async def fetch_multiple_urls(self, urls: list[str]) -> list[str | None]:
        """
        Fetch multiple URLs in parallel

        Args:
            urls: List of URLs to fetch

        Returns:
            List of contents (or None for failed fetches)
        """
        tasks = [self.fetch_page_content(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to None
        contents = []
        for result in results:
            if isinstance(result, Exception):
                contents.append(None)
            else:
                contents.append(result)

        return contents

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Entry point for running the search server"""
    server = SearchServer(max_results=5, fetch_timeout=5)
    await server.run()


if __name__ == "__main__":
    # Run the MCP server
    asyncio.run(main())
