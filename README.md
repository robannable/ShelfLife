# ShelfLife

ShelfLife is an intelligent library cataloguing tool that transforms minimal book input into rich, interconnected bibliographic data. It uses multiple APIs and AI analysis to create a comprehensive personal library management system.

**New:** Refactored with modular architecture, improved error handling, and support for Anthropic Claude and Ollama LLMs.

## Features

- **Minimal data entry** - Just title and author required
- **AI-Powered metadata enhancement** using:
  - Anthropic Claude or Ollama (local LLM)
  - Google Books API
  - Open Library API
- **Cover image handling** with automatic resizing and format conversion
- **Rich book analysis** including:
  - Synopsis generation
  - Theme extraction with hierarchical analysis
  - Genre classification with fiction/non-fiction categorization
  - Historical context
  - Related works suggestions
- **Interactive visualizations**:
  - Genre distribution with sunburst charts
  - Theme relationship analysis
  - Book network connections with force-directed graphs
- **Advanced theme analysis**:
  - Theme extraction and categorization
  - Theme grouping analysis with uber-themes
  - Downloadable theme inventory
- **Search and filter** capabilities
- **Personal notes** for each book
- **Comprehensive analytics** dashboard
- **Executive summary** generation
- **CSV export** functionality
- **Network visualization** showing book relationships

## Architecture

ShelfLife features a modular, maintainable architecture:

### Core Modules

- `shelflife.py` - Streamlit UI and application entry point
- `database.py` - Database operations with error handling and context managers
- `book_service.py` - Book enhancement and LLM integration services
- `llm_client.py` - LLM abstraction layer (Anthropic & Ollama support)
- `models.py` - Data models with validation using dataclasses
- `analytics.py` - Analytics generation and visualization functions
- `api_utils.py` - External API integrations (Google Books, Open Library)
- `validation.py` - **NEW:** Input validation and sanitization
- `logger.py` - Centralized logging system with file rotation
- `constants.py` - Shared constants including genre lists and prompts
- `config.template.py` - **UPDATED:** Environment-based configuration template

### Test Suite

- `tests/test_validation.py` - **NEW:** Validation function tests
- `tests/test_database.py` - **NEW:** Database operation tests
- `tests/test_models.py` - **NEW:** Data model tests
- `tests/test_llm_client.py` - **NEW:** LLM client abstraction tests
- `pytest.ini` - **NEW:** Pytest configuration with coverage settings

### Static Assets

- `static/styles.css` - Modern, responsive styling with dark mode support

### Data Storage

- `data/` - Directory for:
  - SQLite database (`database.db`)
  - Generated JSON catalogs
  - Theme analysis data
  - Executive summaries
  - Log files (`logs/`)

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/ShelfLife.git
cd ShelfLife

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Secure Method)

**⚠️ IMPORTANT: Never commit API keys to version control!**

Create a `.env` file from the example:
```bash
cp .env.example .env
```

Edit `.env` with your actual values:
```bash
# Choose your LLM provider
LLM_PROVIDER=anthropic  # or 'ollama' for local

# Anthropic Configuration (if using Anthropic)
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Google Books API Key (optional but recommended)
GOOGLE_BOOKS_API_KEY=your-google-books-api-key
```

**Security Notes:**
- ✅ `.env` is already in `.gitignore` and won't be committed
- ✅ Never share your `.env` file or commit it to version control
- ✅ Configuration is automatically validated on startup

#### Option A: Anthropic Claude (Recommended)

1. Get an API key at [console.anthropic.com](https://console.anthropic.com/)
2. Set in `.env`:
   ```
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
   ```

#### Option B: Ollama (Local, Free)

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a model: `ollama pull llama3.1`
3. Set in `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.1
   ```

### 3. Optional: Google Books API

For enhanced metadata, get a free API key:
1. Visit [Google Books API](https://developers.google.com/books/docs/v1/using)
2. Add to `.env`:
   ```
   GOOGLE_BOOKS_API_KEY=your-google-books-api-key
   ```

### 4. Run the Application

```bash
streamlit run shelflife.py
```

Open your browser to `http://localhost:8501`

### 5. Run Tests (Optional)

Verify everything is working correctly:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_validation.py -v
```

View coverage report: `open htmlcov/index.html`

## LLM Provider Comparison

| Feature | Anthropic Claude | Ollama |
|---------|-----------------|---------|
| Cost | Pay-per-use | Free (local) |
| Setup | API key only | Install + download models |
| Speed | Fast (cloud) | Depends on hardware |
| Quality | Excellent | Good (model-dependent) |
| Privacy | Cloud-based | Fully local |
| Best for | Production use | Privacy-conscious users |

## Key Improvements (Refactored Version)

### Modular Architecture
- Clean separation of concerns: UI, business logic, data layer
- 1,854-line monolith split into 7 focused modules
- Easy to test, maintain, and extend

### Error Handling
- Comprehensive try-except blocks throughout
- Centralized logging with daily file rotation
- Database context managers for automatic rollback
- Graceful degradation when APIs fail
- User-friendly error messages

### Security & Validation (NEW in v2.0)
- ✅ **SQL Injection Protection:** Whitelisted sort columns with validation
- ✅ **Secure Secrets Management:** Environment variables via `.env` files
- ✅ **Input Validation:** Comprehensive validation for all user inputs
- ✅ **Sanitization:** XSS prevention and injection attack mitigation
- ✅ **Configuration Validation:** Automatic startup checks for required settings
- ✅ **No Hardcoded Secrets:** Template-based configuration with examples

### Testing & Quality (NEW in v2.0)
- ✅ **Comprehensive Test Suite:** 100+ tests covering core functionality
- ✅ **Unit Tests:** Validation, models, database, LLM client
- ✅ **Test Fixtures:** Shared fixtures for consistent testing
- ✅ **Coverage Reporting:** Pytest with coverage tracking
- ✅ **CI-Ready:** Tests can be integrated into GitHub Actions
- 📊 **Target Coverage:** 30-40% initially, expanding to 70%+

### LLM Flexibility
- Support for both cloud (Anthropic) and local (Ollama) LLMs
- Easy to add new LLM providers
- Unified API across providers
- Robust JSON parsing for various response formats

### Performance
- Database connection pooling
- Efficient query optimization
- Caching for API responses
- Batch processing for theme analysis

## Usage

### Adding Books

1. Navigate to "Add Book" page
2. Enter title and author (required)
3. Optionally add: year, ISBN, publisher, condition, cover image, personal notes
4. Click "Add Book" - ShelfLife will automatically fetch and enhance metadata

### Viewing Your Collection

- Search and filter books
- View detailed information including AI-generated analysis
- See related books in your collection
- Export to CSV
- Edit or delete books
- Refresh metadata with latest AI analysis

### Analytics

- View statistics about your collection
- Explore genre distribution (fiction vs. non-fiction)
- Analyze themes across your library
- Extract and group themes into uber-themes
- Download theme inventory

### Network View

- Visualize relationships between books
- Filter by fiction/non-fiction
- See connections based on:
  - Same author
  - Same decade
  - Shared themes

### Executive Summary

- Generate library catalog (JSON)
- Get AI-powered summary of your collection
- Discover patterns in your reading habits
- Receive reading recommendations

### Ask the Library

- Query your collection using natural language
- Get insights about themes, authors, genres
- Discover underrepresented areas
- Get book recommendations based on mood or topic

## Configuration

### Logging

Logs are stored in `data/logs/` with daily rotation.

Adjust log level in `config.py`:
```python
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Debug Mode

Enable detailed error messages:
```python
DEBUG_MODE = True
```

### Image Settings

```python
MAX_IMAGE_SIZE = 800  # pixels
ALLOWED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
```

## Development

### Project Structure
```
ShelfLife/
├── shelflife.py          # Main application
├── database.py           # Database layer
├── book_service.py       # Business logic
├── llm_client.py         # LLM abstraction
├── models.py             # Data models
├── analytics.py          # Analytics functions
├── api_utils.py          # External APIs
├── logger.py             # Logging system
├── constants.py          # Constants
├── config.py             # Configuration (not in repo)
├── config.template.py    # Config template
├── requirements.txt      # Dependencies
├── static/
│   └── styles.css        # Styling
├── data/                 # Data directory (not in repo)
│   ├── database.db
│   ├── logs/
│   └── *.json
└── SETUP_INSTRUCTIONS.md
```

### Adding New Features

- **New LLM provider**: Extend `llm_client.py`
- **New analytics**: Add functions to `analytics.py`
- **New data fields**: Update `models.py` and database schema
- **New UI pages**: Add render functions in `shelflife.py`

### Testing

The modular architecture makes testing easier:

```python
# Example: Test database operations
from database import Database
db = Database(":memory:")  # In-memory database for testing
```

## Troubleshooting

### LLM Connection Issues

1. Check API status using the sidebar button
2. For Anthropic: Verify API key is valid
3. For Ollama: Ensure Ollama is running (`ollama serve`)
4. Check logs in `data/logs/`

### Database Issues

- Ensure `data/` directory is writable
- Check logs for detailed error messages
- Enable `DEBUG_MODE = True` for more information

### Import Errors

```bash
pip install -r requirements.txt --upgrade
```

## Migration from Earlier Versions

The refactored version maintains backward compatibility with existing databases. Your data will be automatically migrated when you first run the new version.

## Requirements

- Python 3.8+
- Streamlit 1.28.0+
- See `requirements.txt` for complete list

## Contributing

Contributions welcome! The modular architecture makes it easy to:
- Add new LLM providers
- Extend analytics
- Add new visualizations
- Improve error handling

## License

[Add your license here]

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- AI powered by [Anthropic Claude](https://www.anthropic.com/) or [Ollama](https://ollama.ai/)
- Metadata from [Google Books](https://books.google.com/) and [Open Library](https://openlibrary.org/)
- Network visualization with [NetworkX](https://networkx.org/) and [Plotly](https://plotly.com/)

## Support

For detailed setup instructions, see [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)

For issues or questions:
- Check the logs in `data/logs/`
- Enable `DEBUG_MODE = True` in config.py
- Review error messages in the Streamlit interface
