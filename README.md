# ShelfLife 📚

ShelfLife is an intelligent library cataloguing tool that transforms minimal book input into rich, interconnected bibliographic data. Perfect for book collectors and personal library management.

## Features

- Minimal data entry requirements (title and author)
- Automatic metadata enhancement using multiple APIs:
  - Open Library
  - Google Books
  - Perplexity AI for deep analysis
- Cover image handling
- Rich book analysis including:
  - Synopsis
  - Themes
  - Genre classification
  - Historical context
  - Related works
- Interactive visualizations:
  - Genre distribution
  - Theme relationships
  - Book network connections
- Search and filter capabilities
- Edit and update functionality
- Analytics dashboard

## Installation

1. Clone the repository
2. Create and activate a virtual environment
3. Install requirements from requirements.txt
4. Create config.py with your API keys:
   - PERPLEXITY_API_KEY
   - GOOGLE_BOOKS_API_KEY
5. Run with: streamlit run shelflife.py

## API Dependencies

- Perplexity AI for enhanced book analysis
- Google Books API for basic metadata
- Open Library API for additional book information

## Development

- Built with Streamlit
- Uses SQLite for local database storage
- Implements caching for API responses
- Includes debug mode for API testing

