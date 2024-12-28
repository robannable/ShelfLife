# API Keys - Replace with your actual keys
PERPLEXITY_API_KEY = "your_perplexity_api_key_here"
GOOGLE_BOOKS_API_KEY = "your_google_books_api_key_here"

# Database configuration
DB_PATH = "database.db"

# API endpoints
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
GOOGLE_BOOKS_API_URL = "https://www.googleapis.com/books/v1/volumes"

# Cache settings
CACHE_EXPIRY = 86400  # 24 hours in seconds

# Image settings
MAX_IMAGE_SIZE = 800  # pixels
ALLOWED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Enhanced analysis prompt
BOOK_ANALYSIS_PROMPT = """
Analyze the following book and return a JSON object with these exact fields:
Book: {title} by {author} ({year})

Return your analysis in this exact JSON format:
{{
    "synopsis": "50-100 word summary",
    "themes": ["theme1", "theme2"],
    "genre": ["primary_genre", "secondary_genre"],
    "historical_context": "brief historical context",
    "related_works": [
        {{
            "title": "Related Book Title",
            "author": "Author Name",
            "reason": "Connection explanation"
        }}
    ],
    "keywords": ["keyword1", "keyword2"],
    "reading_level": "reading level",
    "time_period": "setting time period"
}}

Ensure your response is valid JSON and includes all fields, even if empty.
"""

# Open Library API endpoints
OPEN_LIBRARY_API_URL = "https://openlibrary.org/api/books"
OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"

# Debug mode
DEBUG_MODE = True

# API Test endpoints
API_TEST_ENDPOINTS = {
    "perplexity": {
        "url": "https://api.perplexity.ai/chat/completions",
        "method": "POST",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        "payload": {
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [
                {
                    "role": "user",
                    "content": "Test message"
                }
            ]
        }
    },
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