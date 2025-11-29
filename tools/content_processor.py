"""
Content Processor for Web Search
Handles HTML parsing, content extraction, and snippet generation
"""

import re
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from markdownify import markdownify as md


class ContentProcessor:
    """Process web content for AI consumption"""

    def __init__(self, max_content_length: int = 1500):
        """
        Initialize content processor

        Args:
            max_content_length: Maximum length of extracted content per page (reduced for speed)
        """
        self.max_content_length = max_content_length

    def extract_main_content(self, html: str, url: str = "") -> Optional[str]:
        """
        Extract main content from HTML page

        Args:
            html: Raw HTML content
            url: Source URL (for logging)

        Returns:
            Cleaned text content or None if extraction fails
        """
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer',
                                'aside', 'iframe', 'noscript', 'svg', 'form']):
                element.decompose()

            # Remove elements with common navigation/menu classes
            nav_patterns = ['nav', 'menu', 'sidebar', 'header', 'footer',
                           'advertisement', 'ad-', 'cookie', 'popup', 'modal']
            for pattern in nav_patterns:
                for element in soup.find_all(class_=lambda x: x and pattern in x.lower()):
                    element.decompose()
                for element in soup.find_all(id=lambda x: x and pattern in x.lower()):
                    element.decompose()

            # Try to find main content area (common patterns)
            main_content = None

            # Priority 1: Look for article tags
            if soup.find('article'):
                main_content = soup.find('article')

            # Priority 2: Look for main tag
            elif soup.find('main'):
                main_content = soup.find('main')

            # Priority 3: Look for content-related divs
            elif soup.find(class_=lambda x: x and any(keyword in x.lower()
                                                      for keyword in ['content', 'article', 'post', 'entry'])):
                main_content = soup.find(class_=lambda x: x and any(keyword in x.lower()
                                                                   for keyword in ['content', 'article', 'post', 'entry']))

            # Priority 4: Use body
            else:
                main_content = soup.find('body') or soup

            # Extract text with some structure preservation
            if main_content:
                # Convert to markdown for better structure
                markdown = md(str(main_content), heading_style="ATX")

                # Clean up excessive whitespace
                markdown = re.sub(r'\n\s*\n\s*\n+', '\n\n', markdown)
                markdown = re.sub(r' +', ' ', markdown)
                markdown = markdown.strip()

                # Truncate if too long
                if len(markdown) > self.max_content_length:
                    markdown = markdown[:self.max_content_length] + "..."

                return markdown

            return None

        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return None

    def generate_snippet(self, content: str, query: str = "", max_length: int = 300) -> str:
        """
        Generate a relevant snippet from content

        Args:
            content: Full text content
            query: Search query to find relevant sections
            max_length: Maximum snippet length

        Returns:
            Relevant snippet
        """
        if not content:
            return ""

        # If no query or content is short, return beginning
        if not query or len(content) <= max_length:
            return content[:max_length]

        # Try to find query terms in content
        query_terms = query.lower().split()
        content_lower = content.lower()

        # Find the position with most query term matches
        best_position = 0
        best_score = 0

        # Check every 50 characters
        for i in range(0, len(content) - max_length, 50):
            snippet = content_lower[i:i + max_length]
            score = sum(1 for term in query_terms if term in snippet)
            if score > best_score:
                best_score = score
                best_position = i

        # Extract snippet
        snippet = content[best_position:best_position + max_length]

        # Try to start at sentence boundary
        if best_position > 0:
            sentence_start = snippet.find('. ')
            if sentence_start != -1 and sentence_start < 100:
                snippet = snippet[sentence_start + 2:]

        # Try to end at sentence boundary
        sentence_end = snippet.rfind('. ')
        if sentence_end != -1 and sentence_end > max_length - 100:
            snippet = snippet[:sentence_end + 1]

        # Add ellipsis if truncated
        if best_position > 0:
            snippet = "..." + snippet
        if best_position + len(snippet) < len(content):
            snippet = snippet + "..."

        return snippet.strip()

    def format_search_results(self, results: List[Dict], query: str = "") -> str:
        """
        Format search results concisely for AI consumption

        Args:
            results: List of search results with 'title', 'url', 'content', 'source'
            query: Original search query

        Returns:
            Formatted string with all results (concise version)
        """
        if not results:
            return "No results found."

        formatted = f"Search Results ({len(results)} found):\n\n"

        for i, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            content = result.get('content', '')
            source = result.get('source', 'Unknown')

            # Generate short snippet (reduced from 500 to 300 for speed)
            if len(content) > 300:
                snippet = self.generate_snippet(content, query, max_length=300)
            else:
                snippet = content

            # Format with curly braces for clear structure
            formatted += f"[{i}] {title}\n"
            formatted += f"URL: {url}\n"
            formatted += f"Content: {{{snippet}}}\n"
            formatted += f"Source: {source}\n\n"

        return formatted

    def extract_title(self, html: str) -> str:
        """
        Extract page title from HTML

        Args:
            html: Raw HTML content

        Returns:
            Page title or empty string
        """
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Try <title> tag
            if soup.title and soup.title.string:
                return soup.title.string.strip()

            # Try og:title meta tag
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                return og_title['content'].strip()

            # Try h1 tag
            h1 = soup.find('h1')
            if h1:
                return h1.get_text().strip()

            return ""

        except Exception:
            return ""

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # Normalize quotes
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')

        return text.strip()


def test_content_processor():
    """Test content processor with sample HTML"""
    html = """
    <html>
        <head><title>Test Article</title></head>
        <body>
            <nav>Navigation</nav>
            <article>
                <h1>Main Heading</h1>
                <p>This is the main content of the article. It contains useful information.</p>
                <p>This is more content that should be extracted.</p>
            </article>
            <footer>Footer content</footer>
        </body>
    </html>
    """

    processor = ContentProcessor()
    content = processor.extract_main_content(html)
    print("Extracted content:")
    print(content)
    print("\nSnippet:")
    print(processor.generate_snippet(content or "", "useful information"))


if __name__ == "__main__":
    test_content_processor()
