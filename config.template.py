# LLM Provider Configuration
# Choose 'anthropic' or 'ollama'
LLM_PROVIDER = "anthropic"  # or "ollama" for local models

# Anthropic Configuration (if using Anthropic)
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"  # or "claude-3-opus-20240229", etc.

# Ollama Configuration (if using Ollama)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"  # or any model you have installed

# Google Books API Key (for metadata enhancement)
GOOGLE_BOOKS_API_KEY = "your_google_books_api_key_here"

# Database configuration
DB_PATH = "data/database.db"

# API endpoints
GOOGLE_BOOKS_API_URL = "https://www.googleapis.com/books/v1/volumes"

# Cache settings
CACHE_EXPIRY = 86400  # 24 hours in seconds

# Image settings
MAX_IMAGE_SIZE = 800  # pixels
ALLOWED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

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
DEBUG_MODE = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

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
