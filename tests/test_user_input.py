"""
Unit tests for the user_input module.

Tests cover keyword extraction, semantic matching, and the full ticket
processing pipeline.
"""

import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, '/private/tmp/chuckbot')

from user_input.main import (
    extract_keywords,
    semantic_match,
    process_ticket,
    format_analysis,
    SemanticMatch,
    TicketAnalysis,
    DEFAULT_KEYWORDS,
)


class TestExtractKeywords:
    """Tests for the extract_keywords function."""

    def test_extract_single_keyword(self):
        """Should extract a single matching keyword."""
        content = "I'm having trouble with login"
        keywords = extract_keywords(content)
        assert "login" in keywords

    def test_extract_multiple_keywords(self):
        """Should extract multiple matching keywords."""
        content = "I need help with login and authentication issues"
        keywords = extract_keywords(content)
        assert "login" in keywords
        assert "auth" in keywords

    def test_case_insensitive(self):
        """Should match keywords case-insensitively."""
        content = "LOGIN and AUTHENTICATION problems"
        keywords = extract_keywords(content)
        assert "login" in keywords
        assert "auth" in keywords

    def test_no_matches(self):
        """Should return empty list when no keywords match."""
        content = "This is a random message with no keywords"
        keywords = extract_keywords(content)
        assert keywords == []

    def test_custom_keywords(self):
        """Should include custom keywords in search."""
        content = "I need help with widget configuration"
        keywords = extract_keywords(content, custom_keywords=["widget"])
        assert "widget" in keywords

    def test_empty_content_raises(self):
        """Should raise ValueError for empty content."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            extract_keywords("")

    def test_none_content_raises(self):
        """Should raise ValueError for None content."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            extract_keywords(None)

    def test_removes_duplicates(self):
        """Should not return duplicate keywords."""
        content = "login login login auth auth"
        keywords = extract_keywords(content)
        assert keywords.count("login") == 1
        assert keywords.count("auth") == 1


class TestSemanticMatch:
    """Tests for the semantic_match function."""

    @pytest.fixture
    def document_keys(self):
        """Standard document keys for testing."""
        return [
            "login failure",
            "authentication problem",
            "refund process",
            "deployment issues",
        ]

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock the SentenceTransformer to avoid loading the actual model."""
        with patch('user_input.main.SentenceTransformer') as mock_st:
            with patch('user_input.main.util') as mock_util:
                # Create mock model
                mock_model = MagicMock()
                mock_st.return_value = mock_model

                # Mock embeddings
                mock_model.encode.return_value = MagicMock()

                yield mock_st, mock_util, mock_model

    def test_empty_content_raises(self, document_keys):
        """Should raise ValueError for empty content."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            semantic_match("", document_keys)

    def test_empty_document_keys_raises(self):
        """Should raise ValueError for empty document keys."""
        with pytest.raises(ValueError, match="cannot be None or empty"):
            semantic_match("test content", [])

    def test_invalid_threshold_raises(self, document_keys):
        """Should raise ValueError for invalid threshold."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            semantic_match("test", document_keys, similarity_threshold=1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            semantic_match("test", document_keys, similarity_threshold=-0.1)

    def test_returns_semantic_match_objects(self, document_keys, mock_sentence_transformer):
        """Should return list of SemanticMatch objects."""
        _, mock_util, _ = mock_sentence_transformer

        # Mock similarity scores
        import torch
        mock_similarities = torch.tensor([[0.8, 0.6, 0.3, 0.1]])
        mock_util.pytorch_cos_sim.return_value = mock_similarities

        matches = semantic_match("login issue", document_keys, similarity_threshold=0.2)

        assert all(isinstance(m, SemanticMatch) for m in matches)
        assert len(matches) == 3  # top_k default is 3

    def test_respects_threshold(self, document_keys, mock_sentence_transformer):
        """Should only return matches above threshold."""
        _, mock_util, _ = mock_sentence_transformer

        import torch
        mock_similarities = torch.tensor([[0.8, 0.6, 0.3, 0.1]])
        mock_util.pytorch_cos_sim.return_value = mock_similarities

        matches = semantic_match("login issue", document_keys, similarity_threshold=0.5)

        assert len(matches) == 2  # Only 0.8 and 0.6 are above 0.5
        assert all(m.similarity_score >= 0.5 for m in matches)

    def test_respects_top_k(self, document_keys, mock_sentence_transformer):
        """Should limit results to top_k matches."""
        _, mock_util, _ = mock_sentence_transformer

        import torch
        mock_similarities = torch.tensor([[0.8, 0.7, 0.6, 0.5]])
        mock_util.pytorch_cos_sim.return_value = mock_similarities

        matches = semantic_match(
            "login issue",
            document_keys,
            similarity_threshold=0.0,
            top_k=2
        )

        assert len(matches) == 2

    def test_sorted_by_similarity(self, document_keys, mock_sentence_transformer):
        """Should return matches sorted by similarity (highest first)."""
        _, mock_util, _ = mock_sentence_transformer

        import torch
        mock_similarities = torch.tensor([[0.3, 0.8, 0.5, 0.6]])
        mock_util.pytorch_cos_sim.return_value = mock_similarities

        matches = semantic_match("test", document_keys, similarity_threshold=0.0)

        scores = [m.similarity_score for m in matches]
        assert scores == sorted(scores, reverse=True)


class TestProcessTicket:
    """Tests for the process_ticket function."""

    @pytest.fixture
    def document_keys(self):
        return ["login failure", "authentication problem", "refund process"]

    @pytest.fixture
    def mock_semantic_match(self):
        """Mock semantic_match to avoid loading the model."""
        with patch('user_input.main.semantic_match') as mock:
            mock.return_value = [
                SemanticMatch("login failure", 0.85),
                SemanticMatch("authentication problem", 0.72),
            ]
            yield mock

    def test_returns_ticket_analysis(self, document_keys, mock_semantic_match):
        """Should return TicketAnalysis dataclass."""
        ticket = {
            "id": "001",
            "title": "Test Ticket",
            "content": "I have a login problem"
        }

        result = process_ticket(ticket, document_keys)

        assert isinstance(result, TicketAnalysis)
        assert result.ticket_id == "001"
        assert result.title == "Test Ticket"

    def test_extracts_keywords(self, document_keys, mock_semantic_match):
        """Should extract keywords from content."""
        ticket = {
            "id": "001",
            "title": "Test",
            "content": "Having issues with login and authentication"
        }

        result = process_ticket(ticket, document_keys)

        assert "login" in result.keywords
        assert "auth" in result.keywords

    def test_includes_semantic_matches(self, document_keys, mock_semantic_match):
        """Should include semantic matches in result."""
        ticket = {
            "id": "001",
            "title": "Test",
            "content": "Login problem"
        }

        result = process_ticket(ticket, document_keys)

        assert len(result.semantic_matches) == 2
        assert result.best_match.document_key == "login failure"

    def test_handles_missing_id(self, document_keys, mock_semantic_match):
        """Should use 'unknown' for missing ticket ID."""
        ticket = {"title": "Test", "content": "Login issue"}

        result = process_ticket(ticket, document_keys)

        assert result.ticket_id == "unknown"

    def test_handles_missing_title(self, document_keys, mock_semantic_match):
        """Should use empty string for missing title."""
        ticket = {"id": "001", "content": "Login issue"}

        result = process_ticket(ticket, document_keys)

        assert result.title == ""

    def test_missing_content_raises(self, document_keys):
        """Should raise ValueError when content is missing."""
        ticket = {"id": "001", "title": "Test"}

        with pytest.raises(ValueError, match="must contain 'content' field"):
            process_ticket(ticket, document_keys)

    def test_non_dict_raises(self, document_keys):
        """Should raise ValueError for non-dict ticket_data."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            process_ticket("not a dict", document_keys)

    def test_custom_keywords(self, document_keys, mock_semantic_match):
        """Should pass custom keywords to extract_keywords."""
        ticket = {
            "id": "001",
            "title": "Test",
            "content": "Check out this widget feature"
        }

        result = process_ticket(
            ticket,
            document_keys,
            custom_keywords=["widget"]
        )

        assert "widget" in result.keywords


class TestFormatAnalysis:
    """Tests for the format_analysis function."""

    def test_formats_basic_analysis(self):
        """Should format analysis with all fields."""
        analysis = TicketAnalysis(
            ticket_id="001",
            title="Test Ticket",
            content="Test content",
            keywords=["login", "auth"],
            semantic_matches=[
                SemanticMatch("login failure", 0.85)
            ],
            best_match=SemanticMatch("login failure", 0.85)
        )

        output = format_analysis(analysis)

        assert "Ticket Analysis: 001" in output
        assert "Title: Test Ticket" in output
        assert "Keywords: login, auth" in output
        assert "login failure" in output
        assert "0.850" in output

    def test_handles_empty_keywords(self):
        """Should display 'None' for empty keywords."""
        analysis = TicketAnalysis(
            ticket_id="001",
            title="Test",
            content="Content",
            keywords=[],
            semantic_matches=[],
            best_match=None
        )

        output = format_analysis(analysis)

        assert "Keywords: None" in output

    def test_handles_no_semantic_matches(self):
        """Should handle empty semantic matches."""
        analysis = TicketAnalysis(
            ticket_id="001",
            title="Test",
            content="Content",
            keywords=["test"],
            semantic_matches=[],
            best_match=None
        )

        output = format_analysis(analysis)

        assert "Semantic Matches:" in output
        assert "None" in output


class TestIntegration:
    """Integration tests that use the actual semantic model."""

    @pytest.mark.slow
    def test_end_to_end_processing(self):
        """Full integration test with real model (marked slow)."""
        document_keys = [
            "login failure",
            "authentication problem",
            "refund process",
            "deployment issues"
        ]

        ticket = {
            "id": "integration-001",
            "title": "Can't access my account",
            "content": "I keep getting an error when trying to login to my account"
        }

        result = process_ticket(ticket, document_keys)

        # Should find login-related matches
        assert result.best_match is not None
        assert "login" in result.best_match.document_key.lower() or \
               "auth" in result.best_match.document_key.lower()


# pytest configuration
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
