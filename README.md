# ShelfLife

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

## Project Structure

### Core Files
- `shelflife.py` - Main application file containing the Streamlit interface and core logic
- `config.py` - Configuration file for API keys and application settings
- `api_utils.py` - Utility functions for API interactions and data fetching
- `constants.py` - Shared constants including genre lists, prompts, and taxonomies

### Static Assets
- `static/styles.css` - Custom styling using Swiss Modern design principles
  - Typography using Inter and Space Grotesk
  - Responsive layout with 1200px max width
  - Consistent spacing and visual hierarchy
  - Custom form and interactive element styling

### Data Storage
- `data/` - Directory for storing:
  - SQLite database
  - Generated JSON catalogs
  - Executive summaries
  - Cache files

### Configuration Files
- `.gitignore` - Specifies which files Git should ignore
- `.gitattributes` - Git attributes for file handling
- `requirements.txt` - Python package dependencies
- `setup.sh` - Installation and setup script

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

