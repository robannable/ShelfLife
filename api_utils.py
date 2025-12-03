"""
API utilities for fetching book metadata from external APIs.
Includes retry logic, rate limiting, and caching for reliability and performance.
"""
import requests
import json
from typing import Optional, Dict, Any
import config
from logger import get_logger
from network_utils import (
    retry_with_backoff,
    RetryConfig,
    create_session_with_retries,
    RateLimiter
)
from cache_utils import get_persistent_cache, generate_cache_key

logger = get_logger(__name__)

# Config helper for backward compatibility with older config.py files
def _get_config(attr: str, default: Any) -> Any:
    """Get config attribute with fallback default."""
    return getattr(config, attr, default)

# Default API test endpoints (used if not defined in config.py)
DEFAULT_API_TEST_ENDPOINTS = {
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

# Rate limiters for external APIs
# Google Books: 1000 requests/day (conservative rate limit)
# Open Library: No official limit, but be respectful
GOOGLE_BOOKS_RATE_LIMITER = RateLimiter(calls_per_minute=30, burst_size=10)
OPEN_LIBRARY_RATE_LIMITER = RateLimiter(calls_per_minute=60, burst_size=15)

# Create sessions with retry logic
google_books_session = create_session_with_retries(
    max_retries=2,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504)
)

open_library_session = create_session_with_retries(
    max_retries=2,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504)
)

# Persistent cache for API responses (24 hour TTL)
api_cache = get_persistent_cache()

def test_api_connection(api_name: str) -> Dict[str, bool]:
    """
    Test connection to specified API.
    Returns dict with 'success' boolean and optional 'error' message.
    """
    try:
        endpoint_config = _get_config('API_TEST_ENDPOINTS', DEFAULT_API_TEST_ENDPOINTS).get(api_name)
        if not endpoint_config:
            return {"success": False, "error": f"No configuration found for {api_name}"}
        
        # Get API key if needed
        api_key = None
        if api_name == "google_books":
            api_key = _get_config('GOOGLE_BOOKS_API_KEY', '')
        elif api_name == "perplexity":
            api_key = _get_config('PERPLEXITY_API_KEY', '')
        
        # Prepare request
        url = endpoint_config["url"]
        method = endpoint_config["method"]
        
        # Build request parameters
        kwargs = {}
        if "headers" in endpoint_config:
            kwargs["headers"] = endpoint_config["headers"](api_key) if callable(endpoint_config["headers"]) else endpoint_config["headers"]
        if "params" in endpoint_config:
            kwargs["params"] = endpoint_config["params"](api_key) if callable(endpoint_config["params"]) else endpoint_config["params"]
        if "payload" in endpoint_config:
            kwargs["json"] = endpoint_config["payload"]
        
        # Make request
        response = requests.request(method, url, timeout=5, **kwargs)
        
        # Check response
        if response.status_code == 200:
            return {"success": True}
        else:
            return {
                "success": False,
                "error": f"API returned status code {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"{api_name} API request timed out"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def fetch_book_metadata(title: str, author: str, isbn: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch book metadata using both Open Library and Google Books APIs in a complementary way.
    """
    metadata = {}
    
    # First, try Open Library for basic bibliographic data
    ol_metadata = fetch_from_open_library(title, author, isbn)
    if ol_metadata:
        metadata.update(ol_metadata)
        
    # Then enhance with Google Books data
    gb_metadata = fetch_from_google_books(title, author, isbn)
    if gb_metadata:
        # Prefer Google Books description and cover if available
        if 'description' in gb_metadata:
            metadata['synopsis'] = gb_metadata['description']
        if 'cover_url' in gb_metadata:
            metadata['cover_url'] = gb_metadata['cover_url']
        # Add any additional Google Books data
        metadata.update({
            k: v for k, v in gb_metadata.items() 
            if k not in metadata or not metadata[k]
        })
    
    metadata['sources'] = []
    if ol_metadata:
        metadata['sources'].append('Open Library')
    if gb_metadata:
        metadata['sources'].append('Google Books')
    
    return metadata

@retry_with_backoff(config=RetryConfig(max_retries=2))
def fetch_from_open_library(title: str, author: str, isbn: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata from Open Library API with retry logic, rate limiting, and caching.

    Args:
        title: Book title
        author: Book author
        isbn: Optional ISBN

    Returns:
        Dictionary with book metadata or None if not found
    """
    # Generate cache key
    cache_key = f"open_library:{generate_cache_key(title, author, isbn or '')}"

    # Check cache first
    cached_result = api_cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"Returning cached Open Library result for: {title} by {author}")
        return cached_result

    try:
        # Apply rate limiting
        OPEN_LIBRARY_RATE_LIMITER.wait_if_needed()

        # Try ISBN first if available
        if isbn:
            logger.debug(f"Fetching from Open Library by ISBN: {isbn}")
            response = open_library_session.get(
                f"{_get_config('OPEN_LIBRARY_API_URL', 'https://openlibrary.org/api/books')}?bibkeys=ISBN:{isbn}&format=json&jscmd=data",
                timeout=10
            )
            if response.status_code == 200 and response.json():
                logger.info(f"Found book in Open Library by ISBN: {isbn}")
                result = parse_open_library_response(response.json())
                if result:
                    api_cache.set(cache_key, result, ttl=86400)  # Cache for 24 hours
                return result

        # Fall back to search by title and author
        query = f"title:{title} author:{author}"
        logger.debug(f"Searching Open Library: {query}")
        response = open_library_session.get(
            _get_config('OPEN_LIBRARY_SEARCH_URL', 'https://openlibrary.org/search.json'),
            params={"q": query, "limit": 1},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('docs'):
                logger.info(f"Found book in Open Library: {title} by {author}")
                result = {
                    'title': data['docs'][0].get('title'),
                    'author': data['docs'][0].get('author_name', [author])[0],
                    'year': data['docs'][0].get('first_publish_year'),
                    'publisher': data['docs'][0].get('publisher', [''])[0],
                    'isbn': data['docs'][0].get('isbn', [isbn])[0] if isbn else None,
                    'cover_url': f"https://covers.openlibrary.org/b/id/{data['docs'][0].get('cover_i')}-L.jpg" if data['docs'][0].get('cover_i') else None
                }
                api_cache.set(cache_key, result, ttl=86400)  # Cache for 24 hours
                return result

        logger.warning(f"Book not found in Open Library: {title} by {author}")
        # Cache negative results for shorter time to allow retry
        api_cache.set(cache_key, None, ttl=3600)  # Cache for 1 hour
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Open Library API request error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching from Open Library: {str(e)}", exc_info=True)
        return None

@retry_with_backoff(config=RetryConfig(max_retries=2))
def fetch_from_google_books(title: str, author: str, isbn: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata from Google Books API with retry logic, rate limiting, and caching.

    Args:
        title: Book title
        author: Book author
        isbn: Optional ISBN

    Returns:
        Dictionary with book metadata or None if not found
    """
    # Generate cache key
    cache_key = f"google_books:{generate_cache_key(title, author, isbn or '')}"

    # Check cache first
    cached_result = api_cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"Returning cached Google Books result for: {title} by {author}")
        return cached_result

    try:
        # Apply rate limiting
        GOOGLE_BOOKS_RATE_LIMITER.wait_if_needed()

        # Construct search query
        query = f"intitle:{title} inauthor:{author}"
        if isbn:
            query = f"isbn:{isbn}"

        logger.debug(f"Searching Google Books: {query}")
        response = google_books_session.get(
            _get_config('GOOGLE_BOOKS_API_URL', 'https://www.googleapis.com/books/v1/volumes'),
            params={
                "q": query,
                "key": _get_config('GOOGLE_BOOKS_API_KEY', ''),
                "maxResults": 1
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('items'):
                volume_info = data['items'][0]['volumeInfo']
                logger.info(f"Found book in Google Books: {title} by {author}")
                result = {
                    'title': volume_info.get('title'),
                    'author': volume_info.get('authors', [author])[0],
                    'year': volume_info.get('publishedDate', '')[:4],
                    'publisher': volume_info.get('publisher'),
                    'description': volume_info.get('description'),
                    'cover_url': volume_info.get('imageLinks', {}).get('thumbnail'),
                    'categories': volume_info.get('categories', []),
                    'page_count': volume_info.get('pageCount'),
                    'language': volume_info.get('language')
                }
                api_cache.set(cache_key, result, ttl=86400)  # Cache for 24 hours
                return result

        logger.warning(f"Book not found in Google Books: {title} by {author}")
        # Cache negative results for shorter time to allow retry
        api_cache.set(cache_key, None, ttl=3600)  # Cache for 1 hour
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Google Books API request error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching from Google Books: {str(e)}", exc_info=True)
        return None

def parse_open_library_response(data: Dict) -> Dict[str, Any]:
    """Parse Open Library API response into standardized format."""
    result = {}
    
    if isinstance(data, list):
        data = data[0] if data else {}
    
    result["title"] = data.get("title")
    result["author"] = data.get("author_name", [None])[0]
    result["year"] = data.get("first_publish_year")
    result["publisher"] = data.get("publisher", [None])[0]
    result["isbn"] = data.get("isbn", [None])[0]
    result["language"] = data.get("language", [None])[0]
    result["subjects"] = data.get("subject", [])
    
    # Add cover image URL if available
    if data.get("cover_i"):
        result["cover_url"] = f"https://covers.openlibrary.org/b/id/{data['cover_i']}-L.jpg"
    
    return result 