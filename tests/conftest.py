"""
Shared pytest fixtures and configuration for ShelfLife tests.
"""
import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path so tests can import application modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable config validation during tests
os.environ['SKIP_CONFIG_VALIDATION'] = '1'

# Set test environment variables
os.environ['LLM_PROVIDER'] = 'anthropic'
os.environ['ANTHROPIC_API_KEY'] = 'test-key-for-testing'
os.environ['DEBUG_MODE'] = 'false'
os.environ['LOG_LEVEL'] = 'ERROR'  # Reduce log noise during tests


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def mock_llm_response():
    """Provide a mock LLM response for testing."""
    return '''
{
    "synopsis": "A test book about testing",
    "genre": ["Fiction", "Test"],
    "themes": ["testing", "quality", "automation"],
    "historical_context": "Written in the age of software",
    "reading_level": "Intermediate",
    "time_period": "Modern Era",
    "keywords": ["test", "mock", "fixture"]
}
'''


@pytest.fixture
def sample_library_catalog():
    """Provide a sample library catalog for testing."""
    return {
        "library": [
            {"title": "1984", "author": "George Orwell"},
            {"title": "Brave New World", "author": "Aldous Huxley"},
            {"title": "Fahrenheit 451", "author": "Ray Bradbury"}
        ],
        "generated_at": "2024-01-01T00:00:00"
    }


@pytest.fixture
def sample_themes():
    """Provide sample themes for testing."""
    return [
        "dystopia",
        "totalitarianism",
        "surveillance",
        "censorship",
        "technology",
        "society",
        "freedom",
        "control"
    ]


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield
    # Add cleanup logic here if needed
    pass
