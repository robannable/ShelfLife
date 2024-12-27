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

def fetch_book_metadata(title: str, author: str, isbn: Optional[str] = None) -> Dict[str, Any]:
    """Fetch book metadata from multiple sources."""
    metadata = {"sources": []}
    
    # Try Open Library first
    ol_data = fetch_open_library_data(title, author, isbn)
    if ol_data:
        metadata.update(ol_data)
        metadata["sources"].append("Open Library")
    
    # Try Google Books if needed
    if not metadata.get("year") or not metadata.get("publisher"):
        gb_data = fetch_google_books_data(title, author, isbn)
        if gb_data:
            metadata.update(gb_data)
            metadata["sources"].append("Google Books")
    
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