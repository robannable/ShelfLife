# ShelfLife Setup Instructions

## New Modular Architecture

ShelfLife has been refactored with improved code organization and error handling:

### New Modules
- `logger.py` - Centralized logging system
- `models.py` - Data models and validation
- `database.py` - Database operations with improved error handling
- `llm_client.py` - LLM abstraction layer (Anthropic & Ollama support)
- `book_service.py` - Book enhancement and analysis services
- `analytics.py` - Analytics and visualization functions
- `api_utils.py` - External API integrations (Google Books, Open Library)
- `constants.py` - Shared constants and configurations

### LLM Provider Changes

**Important:** ShelfLife now supports **Anthropic Claude** and **Ollama** instead of Perplexity.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your LLM Provider

Copy the config template:
```bash
cp config.template.py config.py
```

Edit `config.py` and choose your LLM provider:

#### Option A: Use Anthropic Claude (Recommended)

```python
LLM_PROVIDER = "anthropic"
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"  # or another Claude model
```

Get an API key at: https://console.anthropic.com/

#### Option B: Use Ollama (Local, Free)

1. Install Ollama from https://ollama.ai/
2. Pull a model: `ollama pull llama3.1`
3. Configure:

```python
LLM_PROVIDER = "ollama"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"  # or any installed model
```

### 3. Configure Google Books API (Optional but Recommended)

Get a free API key at: https://developers.google.com/books/docs/v1/using

```python
GOOGLE_BOOKS_API_KEY = "your-google-books-api-key"
```

### 4. Run the Application

```bash
streamlit run shelflife.py
```

## Configuration Options

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

Configure maximum image size and allowed types:
```python
MAX_IMAGE_SIZE = 800  # pixels
ALLOWED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
```

## Troubleshooting

### LLM Connection Issues

1. Test your configuration using the "Check API Status" button in the sidebar
2. For Anthropic: Verify your API key is valid
3. For Ollama: Ensure Ollama is running (`ollama serve`)

### Database Issues

If you encounter database errors:
- Check that the `data/` directory is writable
- Logs are in `data/logs/shelflife_YYYYMMDD.log`

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Migration from Old Version

Your existing database will work with the new version. The modular architecture maintains backward compatibility with existing data.

## Features Preserved

All original features are maintained:
- ✅ Book cataloging with minimal data entry
- ✅ AI-powered metadata enhancement
- ✅ Theme analysis and grouping
- ✅ Genre classification
- ✅ Book relationship network visualization
- ✅ Executive summaries
- ✅ CSV export
- ✅ Cover image handling
- ✅ Personal notes

## New Benefits

- 🎯 Better error handling and logging
- 🎯 Modular, maintainable code structure
- 🎯 Support for multiple LLM providers
- 🎯 Improved performance and reliability
- 🎯 Easier testing and debugging

## Support

For issues:
1. Check the logs in `data/logs/`
2. Enable `DEBUG_MODE = True` in config.py
3. Review the error messages in the Streamlit interface

## Development

The refactored codebase is designed for easy extension:
- Add new LLM providers in `llm_client.py`
- Add new analytics in `analytics.py`
- Extend data models in `models.py`
- Add database operations in `database.py`
