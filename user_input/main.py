"""
User Input Processing Module

Handles ticket content extraction, keyword matching, and semantic similarity
matching for the automated response system.
"""

import logging
from typing import Optional
from dataclasses import dataclass

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class SemanticMatch:
    """Represents a semantic match result with document key and similarity score."""
    document_key: str
    similarity_score: float


@dataclass
class TicketAnalysis:
    """Complete analysis result for a processed ticket."""
    ticket_id: str
    title: str
    content: str
    keywords: list[str]
    semantic_matches: list[SemanticMatch]
    best_match: Optional[SemanticMatch]


# Default keywords for extraction - can be extended via configuration
DEFAULT_KEYWORDS = [
    "login", "auth", "authentication", "password", "account",
    "refund", "refunds", "payment", "billing",
    "deployment", "deploy", "preview", "staging",
    "user", "profile", "settings",
    "error", "bug", "issue", "problem", "crash",
    "api", "integration", "webhook"
]


def extract_keywords(ticket_content: str, custom_keywords: Optional[list[str]] = None) -> list[str]:
    """
    Extract known keywords from ticket content.

    Args:
        ticket_content: The text content to analyze
        custom_keywords: Optional list of additional keywords to check

    Returns:
        List of matched keywords found in the content

    Raises:
        ValueError: If ticket_content is None or empty
    """
    if not ticket_content:
        raise ValueError("ticket_content cannot be None or empty")

    keywords = DEFAULT_KEYWORDS.copy()
    if custom_keywords:
        keywords.extend(custom_keywords)

    content_lower = ticket_content.lower()
    extracted = [word for word in keywords if word in content_lower]

    logger.debug(f"Extracted {len(extracted)} keywords from content")
    return list(set(extracted))  # Remove duplicates


def semantic_match(
    ticket_content: str,
    document_keys: list[str],
    similarity_threshold: float = 0.3,
    top_k: int = 3
) -> list[SemanticMatch]:
    """
    Find semantically similar document keys for the given ticket content.

    Uses sentence transformers to compute cosine similarity between the
    ticket content and available document keys. Returns matches above
    the similarity threshold.

    Args:
        ticket_content: The text content to match against
        document_keys: List of document keys/topics to match
        similarity_threshold: Minimum similarity score (0-1) for a match
        top_k: Maximum number of matches to return

    Returns:
        List of SemanticMatch objects sorted by similarity (highest first)

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If model loading or encoding fails
    """
    if not ticket_content:
        raise ValueError("ticket_content cannot be None or empty")
    if not document_keys:
        raise ValueError("document_keys cannot be None or empty")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError as e:
        logger.error("sentence_transformers not installed")
        raise RuntimeError(
            "sentence_transformers package is required. "
            "Install with: pip install sentence-transformers"
        ) from e

    try:
        logger.debug("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        logger.debug("Encoding content and document keys...")
        content_embedding = model.encode(ticket_content, convert_to_tensor=True)
        keys_embedding = model.encode(document_keys, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = util.pytorch_cos_sim(content_embedding, keys_embedding)

    except Exception as e:
        logger.error(f"Error during semantic matching: {e}")
        raise RuntimeError(f"Semantic matching failed: {e}") from e

    # Extract matches above threshold
    matches = []
    if similarities.size(0) > 0:
        similarity_scores = similarities[0].cpu().numpy()

        for idx, score in enumerate(similarity_scores):
            if score >= similarity_threshold:
                matches.append(SemanticMatch(
                    document_key=document_keys[idx],
                    similarity_score=float(score)
                ))

    # Sort by similarity score (descending) and limit to top_k
    matches.sort(key=lambda m: m.similarity_score, reverse=True)
    matches = matches[:top_k]

    logger.debug(f"Found {len(matches)} semantic matches above threshold {similarity_threshold}")
    return matches


def process_ticket(
    ticket_data: dict,
    document_keys: list[str],
    custom_keywords: Optional[list[str]] = None,
    similarity_threshold: float = 0.3
) -> TicketAnalysis:
    """
    Process a support ticket and extract relevant information.

    Performs keyword extraction and semantic matching to categorize
    the ticket and find relevant documentation.

    Args:
        ticket_data: Dictionary with 'id', 'title', and 'content' keys
        document_keys: List of document keys for semantic matching
        custom_keywords: Optional additional keywords to extract
        similarity_threshold: Minimum similarity for semantic matches

    Returns:
        TicketAnalysis dataclass with complete analysis results

    Raises:
        ValueError: If ticket_data is missing required fields
        RuntimeError: If processing fails
    """
    if not isinstance(ticket_data, dict):
        raise ValueError("ticket_data must be a dictionary")

    ticket_id = ticket_data.get("id", "unknown")
    title = ticket_data.get("title", "")
    content = ticket_data.get("content", "")

    if not content:
        raise ValueError("ticket_data must contain 'content' field")

    logger.info(f"Processing ticket {ticket_id}: {title}")

    try:
        # Extract keywords
        keywords = extract_keywords(content, custom_keywords)
        logger.debug(f"Ticket {ticket_id}: Found keywords {keywords}")

        # Perform semantic matching
        semantic_matches = semantic_match(
            content,
            document_keys,
            similarity_threshold=similarity_threshold
        )

        best_match = semantic_matches[0] if semantic_matches else None

        analysis = TicketAnalysis(
            ticket_id=ticket_id,
            title=title,
            content=content,
            keywords=keywords,
            semantic_matches=semantic_matches,
            best_match=best_match
        )

        logger.info(
            f"Ticket {ticket_id} processed: {len(keywords)} keywords, "
            f"{len(semantic_matches)} semantic matches"
        )

        return analysis

    except Exception as e:
        logger.error(f"Failed to process ticket {ticket_id}: {e}")
        raise


def format_analysis(analysis: TicketAnalysis) -> str:
    """Format a TicketAnalysis for display."""
    lines = [
        f"Ticket Analysis: {analysis.ticket_id}",
        f"  Title: {analysis.title}",
        f"  Content: {analysis.content}",
        f"  Keywords: {', '.join(analysis.keywords) if analysis.keywords else 'None'}",
        f"  Semantic Matches:"
    ]

    if analysis.semantic_matches:
        for match in analysis.semantic_matches:
            lines.append(f"    - {match.document_key} (score: {match.similarity_score:.3f})")
    else:
        lines.append("    None")

    if analysis.best_match:
        lines.append(f"  Best Match: {analysis.best_match.document_key}")

    return "\n".join(lines)


def main():
    """Example usage of the ticket processing pipeline."""
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    document_keys = [
        "login failure",
        "authentication problem",
        "refund process",
        "deployment issues",
        "password reset",
        "billing inquiry",
        "api integration help"
    ]

    # Example: Simple query
    simple_ticket = {
        "id": "001",
        "title": "Account Access",
        "content": "Having issues with login and authentication."
    }

    # Example: Complex multi-intent query
    complex_ticket = {
        "id": "002",
        "title": "Multiple Issues",
        "content": "I can't login to my account, and also I need help with a refund for my last payment. Additionally, there seems to be a problem with our API integration."
    }

    print("=" * 60)
    print("Simple Query Example:")
    print("=" * 60)
    analysis = process_ticket(simple_ticket, document_keys)
    print(format_analysis(analysis))

    print("\n" + "=" * 60)
    print("Complex Multi-Intent Query Example:")
    print("=" * 60)
    analysis = process_ticket(complex_ticket, document_keys, similarity_threshold=0.2)
    print(format_analysis(analysis))


if __name__ == "__main__":
    main()
