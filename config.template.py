"""
Configuration management for ShelfLife application.
Uses environment variables for sensitive data with fallbacks for development.
"""
import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, continue with environment variables only
    pass

def get_env_or_raise(key: str, default: Optional[str] = None) -> str:
    """Get environment variable or raise error if not found and no default."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Please set it in your .env file or environment."
        )
    return value

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

# LLM Provider Configuration
# Choose 'anthropic' or 'ollama'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

# Anthropic Configuration (if using Anthropic)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

# Ollama Configuration (if using Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# Google Books API Key (for metadata enhancement)
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "")

# Database configuration
DB_PATH = os.getenv("DB_PATH", "data/database.db")

# API endpoints
GOOGLE_BOOKS_API_URL = "https://www.googleapis.com/books/v1/volumes"

# Cache settings
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "86400"))  # 24 hours in seconds

# Image settings
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "800"))  # pixels
ALLOWED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(5 * 1024 * 1024)))  # 5MB

# Enhanced analysis prompt for LLM
BOOK_ANALYSIS_PROMPT = """Analyze the following book and return a JSON object with comprehensive metadata.

Book: {title} by {author} ({year})

Return your analysis in this exact JSON format:
{{
    "synopsis": "A concise 50-100 word summary of the book",
    "themes": ["theme1", "theme2", "theme3"],
    "genre": ["primary_genre", "secondary_genre"],
    "historical_context": "Brief historical context or significance",
    "related_works": [
        {{
            "title": "Related Book Title",
            "author": "Author Name",
            "reason": "Why this book is related"
        }}
    ],
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "reading_level": "reading level (e.g., Adult, Young Adult, etc.)",
    "time_period": "time period when the book is set or about"
}}

Ensure your response is valid JSON and includes all fields, even if some are empty arrays or null.
"""

# Executive Summary Prompt
EXECUTIVE_SUMMARY_PROMPT = """You are a librarian experienced in curating eclectic book collections. Analyze this library collection and provide a JSON response with the following structure:
{{
    "summary": "A clear paragraph describing the collection's focus and character (approx 200 words). Be abductive and reflect on your chain of thought.",
    "patterns": [
        "Key Pattern 1",
        "Key Pattern 2",
        "Key Pattern 3"
    ],
    "recommendations": [
        "Recommendation 1",
        "Recommendation 2",
        "Recommendation 3"
    ]
}}

Books in collection:
{library_catalog}

Important: Ensure your response contains ONLY valid JSON. Do not include any additional text or formatting."""

# Theme Analysis Prompt
THEME_ANALYSIS_PROMPT = """As a literary analyst, examine these themes and group them into meaningful uber-themes.
Be concise and limit your analysis to the most significant patterns.
Aim to create no more than 10 uber-themes total.

Themes to analyze:
{themes_list}

Respond with a valid JSON object using this exact structure:
{{
    "uber_themes": [
        {{
            "name": "Example Theme Group",
            "description": "Single line description",
            "sub_themes": [
                {{
                    "name": "Original Theme",
                    "connection": "Brief note"
                }}
            ]
        }}
    ],
    "analysis": {{
        "summary": "Brief overview",
        "key_insights": [
            "Key point 1",
            "Key point 2",
            "Key point 3"
        ]
    }}
}}

IMPORTANT FORMATTING RULES:
1. Use double quotes for all strings
2. No trailing commas
3. No line breaks within strings
4. Keep descriptions concise and single-line
5. Ensure all JSON syntax is valid"""

# Open Library API endpoints
OPEN_LIBRARY_API_URL = "https://openlibrary.org/api/books"
OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
OPEN_LIBRARY_COVERS_URL = "https://covers.openlibrary.org/b"

# Google Books settings
GOOGLE_BOOKS_MAX_RESULTS = 1
GOOGLE_BOOKS_FIELDS = "items(volumeInfo(title,authors,publishedDate,description,imageLinks,categories,pageCount,language))"

# Debug and logging
DEBUG_MODE = get_env_bool("DEBUG_MODE", True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# API Test endpoints
API_TEST_ENDPOINTS = {
    "google_books": {
        "url": "https://www.googleapis.com/books/v1/volumes",
        "method": "GET",
        "params": lambda key: {"key": key, "q": "test"}
    },
    "open_library": {
        "url": "https://openlibrary.org/search.json",
        "method": "GET",
        "params": {"q": "test"}
    }
}

# Configuration validation
def validate_config() -> None:
    """
    Validate configuration on application startup.
    Raises ValueError if required configuration is missing or invalid.
    """
    errors = []

    # Validate LLM provider configuration
    if LLM_PROVIDER not in ["anthropic", "ollama"]:
        errors.append(f"Invalid LLM_PROVIDER: '{LLM_PROVIDER}'. Must be 'anthropic' or 'ollama'")

    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required when using Anthropic provider")

    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if LOG_LEVEL.upper() not in valid_log_levels:
        errors.append(f"Invalid LOG_LEVEL: '{LOG_LEVEL}'. Must be one of {valid_log_levels}")

    # Note: Google Books API key is optional (APIs work without it, but with rate limits)
    if not GOOGLE_BOOKS_API_KEY and DEBUG_MODE:
        print("WARNING: GOOGLE_BOOKS_API_KEY is not set. API requests may be rate-limited.")

    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        raise ValueError(error_message)

# Validate on import (can be disabled by setting SKIP_CONFIG_VALIDATION=1)
if not get_env_bool("SKIP_CONFIG_VALIDATION", False):
    try:
        validate_config()
    except ValueError as e:
        # Print error but don't crash on import - let the application handle it
        if DEBUG_MODE:
            print(f"\n⚠️  Configuration Error:\n{e}\n")
            print("Set SKIP_CONFIG_VALIDATION=1 to bypass this check (not recommended)\n")
