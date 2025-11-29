"""
Search Decision Engine
Determines whether a query needs web search or can be answered with general knowledge
"""

import re
from datetime import datetime


class SearchDecisionEngine:
    """
    Intelligent filter to decide if web search is needed
    Prevents unnecessary searches for general knowledge questions
    """

    # Keywords that indicate need for current information
    CURRENT_INFO_KEYWORDS = [
        "latest", "recent", "current", "today", "now", "2024", "2025",
        "this week", "this month", "this year",
        "news", "update", "announcement", "release",
        "breaking", "just", "trending", "viral",
        "price", "stock", "weather", "score", "status",
        "being hyped", "getting popular", "buzz about",
        "everyone talking about"
    ]

    # Keywords that indicate general knowledge (NO search needed)
    GENERAL_KNOWLEDGE_KEYWORDS = [
        "what is", "what are", "define", "definition", "meaning",
        "how does", "how do", "how to", "explain",
        "why is", "why does", "why do",
        "who is", "who was", "who were",
        "when was", "when did",
        "where is", "where are",
        "difference between", "compare",
        "history of", "origin of"
    ]

    # Question types that DON'T need search
    NO_SEARCH_PATTERNS = [
        r"what is \d+\s*[\+\-\*/]\s*\d+",  # Math: "what is 2+2"
        r"calculate",
        r"solve for",
        r"how (old|tall|long|far|many|much)",  # Historical facts
        r"when (did|was) .*(born|die|happen|start|end)",  # Historical dates
        r"who (invented|discovered|created|wrote|painted)",
        r"capital of",
        r"flag of",
        r"population of.*\d{4}",  # Historical population
    ]

    def __init__(self):
        """Initialize decision engine"""
        self.current_year = datetime.now().year

    def needs_search(self, query: str) -> tuple[bool, str]:
        """
        Determine if a query needs web search

        Args:
            query: User's query

        Returns:
            (needs_search: bool, reason: str)
        """
        query_lower = query.lower().strip()

        # Check for explicit search requests
        if query_lower.startswith("search"):
            return True, "Explicit search request"

        # Check for current information keywords
        for keyword in self.CURRENT_INFO_KEYWORDS:
            if keyword in query_lower:
                return True, f"Detected keyword: '{keyword}'"

        # Check if asking about very recent year
        if str(self.current_year) in query or str(self.current_year - 1) in query:
            return True, f"Recent year detected: {self.current_year}"

        # Check for patterns that DON'T need search
        for pattern in self.NO_SEARCH_PATTERNS:
            if re.search(pattern, query_lower):
                return False, f"General knowledge pattern: {pattern}"

        # Check for general knowledge keywords
        for keyword in self.GENERAL_KNOWLEDGE_KEYWORDS:
            if query_lower.startswith(keyword):
                # But check if asking about recent things
                if any(curr in query_lower for curr in ["recent", "latest", "now", "today"]):
                    return True, "Recent information needed despite general format"
                return False, f"General knowledge question: '{keyword}'"

        # Check query length and complexity
        words = query.split()
        if len(words) < 4:
            # Short queries are usually general knowledge
            return False, "Short query - likely general knowledge"

        # Check if asking about concepts vs facts
        concept_indicators = ["concept", "theory", "principle", "algorithm", "method"]
        if any(indicator in query_lower for indicator in concept_indicators):
            return False, "Asking about concepts - general knowledge"

        # Default: For ambiguous cases, DON'T search
        # Better to answer with general knowledge than unnecessary search
        return False, "Ambiguous - defaulting to no search (general knowledge)"

    def explain_decision(self, query: str) -> str:
        """
        Get detailed explanation of search decision

        Args:
            query: User's query

        Returns:
            Explanation string
        """
        needs, reason = self.needs_search(query)

        if needs:
            return f"🔍 Web search needed: {reason}"
        else:
            return f"💡 No search needed: {reason}"


# Quick test
def test_decision_engine():
    """Test the decision engine with various queries"""
    engine = SearchDecisionEngine()

    test_cases = [
        # Should NOT search (general knowledge)
        "What is Python?",
        "How does photosynthesis work?",
        "When did World War 2 end?",
        "Who invented the telephone?",
        "What is 2 + 2?",
        "Explain quantum mechanics",
        "Define recursion",
        "What is the capital of France?",

        # SHOULD search (current info)
        "Latest AI developments",
        "Why is Claude being hyped?",
        "What happened today?",
        "Python 3.12 release date",
        "Current weather in New York",
        "Bitcoin price",
        "Latest news about SpaceX",
        "What's trending on Twitter?",
    ]

    print("="*80)
    print("SEARCH DECISION ENGINE TEST")
    print("="*80)

    for query in test_cases:
        needs, reason = engine.needs_search(query)
        status = "🔍 SEARCH" if needs else "💡 NO SEARCH"
        print(f"\n{status}: {query}")
        print(f"   Reason: {reason}")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_decision_engine()
