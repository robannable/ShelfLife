# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ShelfLife is an intelligent library cataloguing tool that transforms minimal book input into rich, interconnected bibliographic data using AI. It's a Streamlit web application built with a modular architecture supporting multiple LLM providers (Anthropic Claude, Ollama).

## Development Commands

### Running the Application
```bash
streamlit run shelflife.py
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_validation.py -v

# Run specific test markers
pytest -m unit           # Unit tests only
pytest -m database       # Database tests only
pytest -m api           # API tests (may make real calls)
```

### Setup
```bash
# First-time setup
cp config.template.py config.py    # Then edit config.py with your settings
cp .env.example .env              # Then edit .env with API keys
pip install -r requirements.txt

# Or use the setup script
chmod +x setup.sh
./setup.sh
```

## Architecture

### Core Design Patterns

**Singleton Services**: Database and BookService are initialized once using Streamlit's `@st.cache_resource` decorator:
```python
@st.cache_resource
def get_database():
    return Database(config.DB_PATH)
```

**Context Managers for Database**: All database operations use context managers for automatic commit/rollback:
```python
with self.get_connection() as conn:
    # Database operations here
    # Auto-commits on success, auto-rolls back on exception
```

**LLM Provider Abstraction**: The `LLMClient` abstract base class allows easy swapping between providers. Use `LLMClientFactory.create_client()` to instantiate:
```python
# In book_service.py
self.llm_client = LLMClientFactory.create_client(
    provider="anthropic",
    api_key=config.ANTHROPIC_API_KEY,
    model=config.ANTHROPIC_MODEL
)
```

**Network Resilience**: All external API calls use retry logic, rate limiting, and circuit breakers via `network_utils.py`:
- `@retry_with_backoff()` decorator for automatic retries
- `RateLimiter` class for rate limiting (50 req/min for Anthropic, 30 for Google Books)
- `CircuitBreaker` class to prevent cascading failures
- `create_session_with_retries()` for requests session with built-in retry strategy

### Data Flow

1. **User Input** → `shelflife_app/views/*.py` (UI layer, one script per page)
2. **Input Validation** → `validation.py` (sanitization, XSS prevention)
3. **Book Enhancement** → `book_service.py` → `api_utils.py` + `llm_client.py`
4. **Data Storage** → `database.py` (with context managers)
5. **Analytics** → `analytics.py` (generates visualizations)

### Module Responsibilities

**Entry & UI package**
- **shelflife.py**: Thin entry point. Configures `st.set_page_config`, loads CSS, mounts `st.navigation` with the six page scripts, and renders the sidebar extras.
- **shelflife_app/services.py**: `get_database()` and `get_book_service()` singletons (via `@st.cache_resource`) plus `_get_config()` helper.
- **shelflife_app/ui_helpers.py**: Shared UI primitives (`render_page_header`, `render_empty_state`, `render_condition_badge`, `render_tags`) and visual constants (`CHART_COLORS`, `CONDITION_BADGES`).
- **shelflife_app/image_utils.py**: `process_image()` for cover uploads.
- **shelflife_app/library_utils.py**: Library-wide helpers (`generate_library_json`, `find_related_books`, `display_theme_analysis`).
- **shelflife_app/sidebar.py**: `render_sidebar_extras()` — logo, API status check, library stats.
- **shelflife_app/views/**: One script per navigable page (`add_book.py`, `view_collection.py`, `analytics.py`, `network_view.py`, `executive_summary.py`, `ask_library.py`). Each script is executed by `st.navigation` and uses top-level Streamlit calls (no `def main()` wrapper).

**Backend**
- **database.py**: All SQLite operations, schema versioning, backup/restore utilities
- **book_service.py**: Book metadata enhancement orchestration, calls APIs and LLM
- **llm_client.py**: LLM provider abstraction (AnthropicClient, OllamaClient)
- **models.py**: Dataclasses for Book, BookMetadata, LibraryStats with validation in `__post_init__`
- **analytics.py**: Statistics generation, network graph creation, visualization functions
- **api_utils.py**: External API integrations (Google Books, Open Library) with retry logic
- **network_utils.py**: Retry logic, rate limiting, circuit breaker pattern
- **validation.py**: Input validation, sanitization, XSS prevention
- **logger.py**: Centralized logging with daily rotation
- **constants.py**: Genre lists, prompts, configuration constants
- **config.template.py**: Configuration template (copy to config.py)

### Page Navigation

The app uses Streamlit's modern multipage pattern (`st.Page` + `st.navigation`, requires Streamlit ≥ 1.36). Navigation lives in the sidebar and is constructed in `shelflife.py:main()`. To switch pages programmatically from a view, call `st.switch_page("shelflife_app/views/<name>.py")`.

To add a new page:
1. Create `shelflife_app/views/new_page.py` with top-level Streamlit code (no function wrapper). Use `get_database()` / `get_book_service()` from `shelflife_app.services` for singletons.
2. Append an `st.Page(...)` entry to the `pages` list in `shelflife.py`.

### Database Schema

**Schema Versioning**: The database uses a version table and applies migrations incrementally in `_init_schema()`.

**Main Table**: `books` table with columns:
- Core: id, title, author, year, isbn, publisher, condition
- Media: cover_image (BLOB)
- Rich Data: metadata (JSON), personal_notes (TEXT)
- Timestamps: created_at, updated_at

**Metadata JSON Structure**: Stored in the metadata column as JSON:
```json
{
  "synopsis": "...",
  "genre": ["Fiction", "Historical Fiction"],
  "themes": ["War", "Love", "Loss"],
  "historical_context": "...",
  "time_period": "1940s",
  "related_works": [{"title": "...", "author": "...", "reason": "..."}],
  "cover_url": "https://...",
  "sources": ["google_books", "llm"]
}
```

### Configuration Management

**Environment Variables**: All sensitive configuration uses `.env` file (via python-dotenv):
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_BOOKS_API_KEY=...
```

**Config Loading**: `config.template.py` loads from environment with helpers:
- `get_env_or_raise()`: Required values, raises if missing
- `get_env_bool()`: Boolean parsing
- Direct `os.getenv()`: Optional values with defaults

**Important**: Never commit `config.py` or `.env` files (both in `.gitignore`).

## Testing Strategy

### Test Structure
```
tests/
├── conftest.py              # Shared fixtures (sample_book, mock LLM responses)
├── test_validation.py       # Input validation and sanitization
├── test_models.py          # Data model validation
├── test_database.py        # Database operations and backup/restore
├── test_llm_client.py      # LLM client abstraction
└── test_network_utils.py   # Retry logic, rate limiting, circuit breaker
```

### Key Fixtures (conftest.py)
- `sample_book()`: Returns a valid Book instance
- `sample_metadata()`: Returns BookMetadata with typical data
- `temp_db()`: Temporary SQLite database for testing
- `mock_anthropic_response()`: Mock LLM API responses

### Testing Guidelines
- Database tests use in-memory SQLite (`:memory:`)
- Mock external API calls to avoid rate limits
- Mark API tests with `@pytest.mark.api` for conditional running
- Coverage target: 70%+ (currently 40-50%)

## Common Development Patterns

### Adding a New Book Field

1. Update `models.py` Book dataclass:
```python
@dataclass
class Book:
    new_field: Optional[str] = None
```

2. Add database migration in `database.py`:
```python
schema_updates = {
    3: '''
        ALTER TABLE books ADD COLUMN new_field TEXT;
    '''
}
```

3. Update the relevant view script in `shelflife_app/views/` (typically `add_book.py` and `view_collection.py`) to collect/display the field

### Adding a New LLM Provider

1. Create new client class in `llm_client.py`:
```python
class NewProviderClient(LLMClient):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        # Implementation
        pass

    def test_connection(self) -> Dict[str, Any]:
        # Implementation
        pass
```

2. Register in `LLMClientFactory.create_client()`

3. Add configuration to `config.template.py`

### Adding a New Analytics Function

1. Add function to `analytics.py`:
```python
def new_analysis(books_data: List[Tuple]) -> pd.DataFrame:
    # Process books_data
    return results_df
```

2. Call from the appropriate view script (e.g. `shelflife_app/views/analytics.py`):
```python
results = new_analysis(books)
st.plotly_chart(create_visualization(results))
```

## Important Implementation Details

### JSON Parsing from LLM Responses

LLMs may return JSON wrapped in markdown code blocks. Use `parse_json_from_response()` from `llm_client.py`:
```python
from llm_client import parse_json_from_response

response = llm_client.generate(prompt)
data = parse_json_from_response(response)
```

### Image Processing

Images are stored as BLOBs in the database. Use `process_image()` from `shelflife_app.image_utils`:
- Automatically resizes to MAX_IMAGE_SIZE (800px)
- Converts to appropriate format (JPEG, PNG, etc.)
- Returns bytes for database storage

### Input Validation

Always validate user input before processing:
```python
from validation import validate_book_input, sanitize_string

is_valid, errors = validate_book_input(title, author, year, isbn)
if not is_valid:
    st.error(f"Validation errors: {', '.join(errors.values())}")
    return
```

### Database Backup Best Practices

Use built-in backup utilities before destructive operations:
```python
db = get_database()

# Create timestamped backup
backup_path = db.backup_database()

# Restore from backup (with confirmation)
db.restore_database(backup_path, confirm=True)

# Check database health
health = db.health_check()
```

### Rate Limiting for External APIs

Rate limiters are pre-configured in `llm_client.py`:
- Anthropic: 50 requests/minute
- Ollama: 120 requests/minute (local, but limited for safety)
- Google Books: 30 requests/minute (configured in `api_utils.py`)

Use `RateLimiter.wait_if_needed()` before API calls.

## Streamlit-Specific Patterns

### Session State for Multi-Page State
```python
if 'selected_book_id' not in st.session_state:
    st.session_state.selected_book_id = None
```

### Caching Expensive Operations
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def expensive_analysis(books):
    # Heavy processing
    return results
```

### Progress Indicators for Long Operations
```python
with st.spinner("Enhancing book metadata..."):
    metadata = book_service.enhance_book_data(title, author)
```

## Logging

All modules use centralized logging via `logger.py`:
```python
from logger import get_logger

logger = get_logger(__name__)
logger.info("Processing book: {title}")
logger.error(f"Failed to fetch metadata: {str(e)}", exc_info=True)
```

Logs are stored in `data/logs/` with daily rotation. Log level configured in `config.py`.

## Genre Classification System

The application uses a two-tier genre system defined in `constants.py`:

**GENRE_CATEGORIES**: Fiction vs. Non-fiction classification
**STANDARD_GENRES**: Detailed genre list constraining LLM responses

When prompting LLMs for genre classification, always include the GENRE_PROMPT to ensure consistency.

## Security Considerations

- **SQL Injection**: All database queries use parameterized statements
- **XSS Prevention**: User input sanitized via `validation.sanitize_string()`
- **API Key Security**: All keys in `.env` file (never committed)
- **Input Validation**: Whitelisted characters and length limits
- **Sort Column Whitelisting**: Only allowed column names in database queries

## Known Limitations

- Images stored as BLOBs (may want to use file storage for very large collections)
- Theme analysis processes all books at once (may need batching for 10,000+ books)
- Circuit breaker state is not persisted across app restarts
- No user authentication (single-user application)
