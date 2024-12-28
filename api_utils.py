import requests
import json
import streamlit as st
from typing import Dict, Any, Optional
import config

def test_api_connection(api_name: str) -> Dict[str, Any]:
    """Test connection to specified API and return status information."""
    api_config = config.API_TEST_ENDPOINTS.get(api_name)
    if not api_config:
        return {"success": False, "error": f"No configuration found for {api_name}"}
    
    try:
        if api_name == "perplexity":
            headers = api_config["headers"](config.PERPLEXITY_API_KEY)
            response = requests.post(
                api_config["url"],
                headers=headers,
                json=api_config["payload"],
                timeout=15
            )
        elif api_name == "google_books":
            params = api_config["params"](config.GOOGLE_BOOKS_API_KEY)
            response = requests.get(
                api_config["url"],
                params=params,
                timeout=5
            )
        else:  # open_library
            response = requests.get(
                api_config["url"],
                params=api_config["params"],
                timeout=5
            )
        
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else None,
            "error": None if response.status_code == 200 else response.text
        }
    
    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "response": None,
            "error": str(e)
        }

def fetch_book_metadata(title: str, author: str, isbn: Optional[str] = None) -> dict:
    """Fetch book metadata from multiple sources, prioritizing ISBN when available."""
    metadata = {"sources": []}

    # Try ISBN-based lookup first if available
    if isbn:
        # Google Books ISBN search
        try:
            google_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&key={config.GOOGLE_BOOKS_API_KEY}"
            response = requests.get(google_url)
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    book_info = data['items'][0]['volumeInfo']
                    metadata.update({
                        "title": book_info.get('title'),
                        "author": book_info.get('authors', [author])[0],
                        "year": int(book_info.get('publishedDate', '').split('-')[0]) if book_info.get('publishedDate') else None,
                        "publisher": book_info.get('publisher'),
                        "description": book_info.get('description'),
                        "cover_url": book_info.get('imageLinks', {}).get('thumbnail'),
                        "categories": book_info.get('categories', [])
                    })
                    metadata["sources"].append("Google Books (ISBN)")
        except Exception as e:
            if config.DEBUG_MODE:
                st.error(f"Google Books ISBN API error: {str(e)}")

        # Open Library ISBN search
        try:
            ol_url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
            response = requests.get(ol_url)
            if response.status_code == 200:
                data = response.json()
                if data:
                    book_info = data.get(f"ISBN:{isbn}", {})
                    if book_info:
                        metadata.update({
                            "ol_subjects": book_info.get('subjects', []),
                            "ol_cover_url": book_info.get('cover', {}).get('large'),
                            "ol_publish_date": book_info.get('publish_date')
                        })
                        metadata["sources"].append("Open Library (ISBN)")
        except Exception as e:
            if config.DEBUG_MODE:
                st.error(f"Open Library ISBN API error: {str(e)}")

    # Fallback to title/author search if ISBN search failed or wasn't available
    if "Google Books" not in metadata["sources"]:
        try:
            google_url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title}+inauthor:{author}&key={config.GOOGLE_BOOKS_API_KEY}"
            response = requests.get(google_url)
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    book_info = data['items'][0]['volumeInfo']
                    metadata.update({
                        "title": book_info.get('title'),
                        "author": book_info.get('authors', [author])[0],
                        "year": int(book_info.get('publishedDate', '').split('-')[0]) if book_info.get('publishedDate') else None,
                        "publisher": book_info.get('publisher'),
                        "description": book_info.get('description'),
                        "cover_url": book_info.get('imageLinks', {}).get('thumbnail'),
                        "categories": book_info.get('categories', [])
                    })
                    metadata["sources"].append("Google Books")
        except Exception as e:
            if config.DEBUG_MODE:
                st.error(f"Google Books API error: {str(e)}")

    if "Open Library" not in metadata["sources"]:
        try:
            ol_url = f"https://openlibrary.org/search.json?title={title}&author={author}"
            response = requests.get(ol_url)
            if response.status_code == 200:
                data = response.json()
                if data.get('docs'):
                    book_info = data['docs'][0]
                    metadata.update({
                        "ol_subjects": book_info.get('subject', []),
                        "ol_first_publish_year": book_info.get('first_publish_year'),
                        "ol_cover_id": book_info.get('cover_i')
                    })
                    if book_info.get('cover_i'):
                        metadata["ol_cover_url"] = f"https://covers.openlibrary.org/b/id/{book_info['cover_i']}-L.jpg"
                    metadata["sources"].append("Open Library")
        except Exception as e:
            if config.DEBUG_MODE:
                st.error(f"Open Library API error: {str(e)}")

    return metadata

def fetch_open_library_data(title: str, author: str, isbn: Optional[str] = None) -> Dict[str, Any]:
    """Fetch book data from Open Library API."""
    try:
        # Try ISBN first if available
        if isbn:
            response = requests.get(
                f"{config.OPEN_LIBRARY_API_URL}?bibkeys=ISBN:{isbn}&format=json&jscmd=data",
                timeout=5
            )
            if response.status_code == 200 and response.json():
                return parse_open_library_response(response.json())
        
        # Search by title and author if ISBN search fails or isn't available
        query = f"title:{title} author:{author}"
        response = requests.get(
            config.OPEN_LIBRARY_SEARCH_URL,
            params={"q": query},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("docs"):
                return parse_open_library_response(data["docs"][0])
    
    except Exception as e:
        if config.DEBUG_MODE:
            st.error(f"Open Library API error: {str(e)}")
        return None
    
    return None

def fetch_google_books_data(title: str, author: str, isbn: Optional[str] = None) -> Dict[str, Any]:
    """Fetch book data from Google Books API."""
    try:
        query = f"intitle:{title} inauthor:{author}"
        if isbn:
            query += f" isbn:{isbn}"
            
        response = requests.get(
            config.GOOGLE_BOOKS_API_URL,
            params={
                "q": query,
                "key": config.GOOGLE_BOOKS_API_KEY
            },
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("items"):
                return parse_google_books_response(data["items"][0])
    
    except Exception as e:
        if config.DEBUG_MODE:
            st.error(f"Google Books API error: {str(e)}")
        return None
    
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

def parse_google_books_response(data: Dict) -> Dict[str, Any]:
    """Parse Google Books API response into standardized format."""
    volume_info = data.get("volumeInfo", {})
    result = {}
    
    result["title"] = volume_info.get("title")
    result["author"] = volume_info.get("authors", [None])[0]
    result["year"] = volume_info.get("publishedDate", "")[:4]
    result["publisher"] = volume_info.get("publisher")
    result["isbn"] = next((i["identifier"] for i in volume_info.get("industryIdentifiers", []) 
                          if i["type"] in ["ISBN_13", "ISBN_10"]), None)
    result["language"] = volume_info.get("language")
    result["subjects"] = volume_info.get("categories", [])
    
    # Add cover image URL if available
    if "imageLinks" in volume_info:
        result["cover_url"] = volume_info["imageLinks"].get("thumbnail", "").replace("http://", "https://")
    
    return result 