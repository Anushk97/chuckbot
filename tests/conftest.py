"""
Pytest configuration and fixtures for chuckbot tests.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def sample_ticket():
    """Provide a sample ticket for testing."""
    return {
        "id": "test-001",
        "title": "Test Ticket",
        "content": "This is test content for the ticket."
    }


@pytest.fixture
def sample_document_keys():
    """Provide sample document keys for testing."""
    return [
        "login failure",
        "authentication problem",
        "refund process",
        "deployment issues",
        "password reset",
        "billing inquiry"
    ]
