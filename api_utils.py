import requests
import json
from typing import Optional, Dict, Any
import config

def test_api_connection(api_name: str) -> Dict[str, bool]:
    """
    Test connection to specified API.
    Returns dict with 'success' boolean and optional 'error' message.
    """
    try:
        endpoint_config = config.API_TEST_ENDPOINTS.get(api_name)
        if not endpoint_config:
            return {"success": False, "error": f"No configuration found for {api_name}"}
        
        # Get API key if needed
        api_key = None
        if api_name == "google_books":
            api_key = config.GOOGLE_BOOKS_API_KEY
        elif api_name == "perplexity":
            api_key = config.PERPLEXITY_API_KEY
        
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

def fetch_from_open_library(title: str, author: str, isbn: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata from Open Library API.
    """
    try:
        # Try ISBN first if available
        if isbn:
            response = requests.get(
                f"{config.OPEN_LIBRARY_API_URL}?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
            )
            if response.status_code == 200 and response.json():
                return parse_open_library_response(response.json())
        
        # Fall back to search by title and author
        query = f"title:{title} author:{author}"
        response = requests.get(
            config.OPEN_LIBRARY_SEARCH_URL,
            params={"q": query, "limit": 1}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('docs'):
                return {
                    'title': data['docs'][0].get('title'),
                    'author': data['docs'][0].get('author_name', [author])[0],
                    'year': data['docs'][0].get('first_publish_year'),
                    'publisher': data['docs'][0].get('publisher', [''])[0],
                    'isbn': data['docs'][0].get('isbn', [isbn])[0] if isbn else None,
                    'cover_url': f"https://covers.openlibrary.org/b/id/{data['docs'][0].get('cover_i')}-L.jpg" if data['docs'][0].get('cover_i') else None
                }
        
        return None
            
    except Exception as e:
        if config.DEBUG_MODE:
            print(f"Open Library API error: {str(e)}")
        return None

def fetch_from_google_books(title: str, author: str, isbn: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata from Google Books API.
    """
    try:
        # Construct search query
        query = f"intitle:{title} inauthor:{author}"
        if isbn:
            query = f"isbn:{isbn}"
            
        response = requests.get(
            config.GOOGLE_BOOKS_API_URL,
            params={
                "q": query,
                "key": config.GOOGLE_BOOKS_API_KEY,
                "maxResults": 1
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('items'):
                volume_info = data['items'][0]['volumeInfo']
                return {
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
        
        return None
            
    except Exception as e:
        if config.DEBUG_MODE:
            print(f"Google Books API error: {str(e)}")
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