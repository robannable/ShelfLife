# ShelfLife

ShelfLife is an intelligent library cataloguing tool that transforms minimal book input into rich, interconnected bibliographic data. It uses multiple APIs and AI analysis to create a comprehensive personal library management system.

## Features

- Minimal data entry requirements (title and author)
- Automatic metadata enhancement using multiple APIs:
  - Open Library
  - Google Books
  - Perplexity AI for deep analysis
- Cover image handling with automatic resizing and format conversion
- Rich book analysis including:
  - Synopsis
  - Themes with hierarchical analysis
  - Genre classification with fiction/non-fiction categorization
  - Historical context
  - Related works suggestions
- Interactive visualizations:
  - Genre distribution with sunburst charts
  - Theme relationship analysis
  - Book network connections with force-directed graphs
- Advanced theme analysis:
  - Theme extraction and categorization
  - Theme grouping analysis
  - Downloadable theme inventory
- Search and filter capabilities
- Edit and update functionality
- Comprehensive analytics dashboard
- Executive summary generation
- CSV export functionality

## Project Structure

### Core Files
- `shelflife.py` - Main application file containing the Streamlit interface and core logic
- `config.py` - Configuration file for API keys and application settings
- `api_utils.py` - Utility functions for API interactions and data fetching
- `constants.py` - Shared constants including genre lists, prompts, and taxonomies

### Static Assets
- `static/styles.css`
  - Modern typography using Inter and Space Grotesk
  - Responsive layout with 1200px max width
  - Dark mode compatible styling
  - Custom form and interactive element styling

### Data Storage
- `data/` - Directory for storing:
  - SQLite database
  - Generated JSON catalogs
  - Theme analysis data
  - Executive summaries
  - Cache files

### Configuration Files
- `.gitignore` - Specifies which files Git should ignore
- `.gitattributes` - Git attributes for file handling
- `requirements.txt` - Python package dependencies
- `setup.sh` - Linux/macOS installation script
- `setup.bat` - Windows installation script

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
   - Windows: Run `setup.bat`
   - Linux/macOS: Run `./setup.sh`
3. Copy `config.template.py` to `config.py` and add your API keys:
   - PERPLEXITY_API_KEY
   - GOOGLE_BOOKS_API_KEY
4. Run with: `streamlit run shelflife.py`

## API Dependencies

- Perplexity AI for enhanced book analysis and theme extraction
- Google Books API for basic metadata
- Open Library API for additional book information

## Development

- Built with Streamlit for rapid deployment
- Uses SQLite for efficient local database storage
- Implements LRU caching for API responses
- Includes debug mode for API testing
- Supports batch processing for theme analysis
- Modular design for easy feature expansion

## Data Analysis Features

- Genre distribution analysis with fiction/non-fiction categorization
- Theme extraction and hierarchical analysis
- Network analysis of book relationships
- Executive summary generation
- Exportable data in JSON and CSV formats