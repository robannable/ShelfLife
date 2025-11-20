"""
Input validation and sanitization utilities for ShelfLife.
"""
import re
from typing import Optional, Tuple
from datetime import datetime
from logger import get_logger

logger = get_logger(__name__)

# Input length constraints
MAX_TITLE_LENGTH = 500
MAX_AUTHOR_LENGTH = 200
MAX_ISBN_LENGTH = 20
MAX_PUBLISHER_LENGTH = 200
MAX_SEARCH_LENGTH = 200
MAX_NOTES_LENGTH = 10000

# Allowed characters patterns
ALLOWED_TITLE_PATTERN = re.compile(r'^[\w\s\-\'\",.:;!?()\[\]]+$', re.UNICODE)
ALLOWED_AUTHOR_PATTERN = re.compile(r'^[\w\s\-\'.]+$', re.UNICODE)
ALLOWED_ISBN_PATTERN = re.compile(r'^[\dX\-]+$')  # ISBN can contain digits, X, and hyphens


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def sanitize_string(text: str, max_length: int) -> str:
    """
    Sanitize a string by trimming whitespace and limiting length.

    Args:
        text: Input string to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string
    """
    if not text:
        return ""

    # Strip leading/trailing whitespace
    text = text.strip()

    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    # Truncate if too long
    if len(text) > max_length:
        logger.warning(f"Input truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]

    return text


def validate_title(title: str) -> Tuple[bool, Optional[str]]:
    """
    Validate book title.

    Args:
        title: Book title to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not title or not title.strip():
        return False, "Title is required and cannot be empty"

    title = sanitize_string(title, MAX_TITLE_LENGTH)

    if len(title) < 1:
        return False, "Title must be at least 1 character long"

    if len(title) > MAX_TITLE_LENGTH:
        return False, f"Title must be less than {MAX_TITLE_LENGTH} characters"

    # Allow most Unicode characters for international titles
    # Just block control characters and common injection patterns
    if any(ord(c) < 32 for c in title if c not in '\n\r\t'):
        return False, "Title contains invalid control characters"

    return True, None


def validate_author(author: str) -> Tuple[bool, Optional[str]]:
    """
    Validate author name.

    Args:
        author: Author name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not author or not author.strip():
        return False, "Author is required and cannot be empty"

    author = sanitize_string(author, MAX_AUTHOR_LENGTH)

    if len(author) < 1:
        return False, "Author name must be at least 1 character long"

    if len(author) > MAX_AUTHOR_LENGTH:
        return False, f"Author name must be less than {MAX_AUTHOR_LENGTH} characters"

    # Allow most Unicode characters for international names
    if any(ord(c) < 32 for c in author if c not in '\n\r\t'):
        return False, "Author name contains invalid control characters"

    return True, None


def validate_year(year: Optional[int]) -> Tuple[bool, Optional[str]]:
    """
    Validate publication year.

    Args:
        year: Publication year to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if year is None:
        return True, None  # Year is optional

    current_year = datetime.now().year

    if not isinstance(year, int):
        return False, "Year must be an integer"

    if year < -3000:  # Arbitrary ancient date
        return False, "Year seems too old to be valid"

    if year > current_year + 5:  # Allow some future publications
        return False, f"Year cannot be more than 5 years in the future (current: {current_year})"

    return True, None


def validate_isbn(isbn: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate ISBN.

    Args:
        isbn: ISBN to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isbn or not isbn.strip():
        return True, None  # ISBN is optional

    isbn = sanitize_string(isbn, MAX_ISBN_LENGTH)

    # Remove common separators
    isbn_clean = isbn.replace('-', '').replace(' ', '')

    # ISBN-10 or ISBN-13
    if len(isbn_clean) not in [10, 13]:
        return False, "ISBN must be 10 or 13 characters (excluding hyphens)"

    # Check if contains only digits and optionally 'X' at the end for ISBN-10
    if not ALLOWED_ISBN_PATTERN.match(isbn):
        return False, "ISBN can only contain digits, 'X', and hyphens"

    return True, None


def validate_search_term(search_term: str) -> Tuple[bool, Optional[str]]:
    """
    Validate search term.

    Args:
        search_term: Search term to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not search_term:
        return True, None  # Empty search is valid (returns all)

    search_term = sanitize_string(search_term, MAX_SEARCH_LENGTH)

    if len(search_term) > MAX_SEARCH_LENGTH:
        return False, f"Search term must be less than {MAX_SEARCH_LENGTH} characters"

    # Block potential SQL injection patterns
    dangerous_patterns = [
        r';\s*drop\s+table',
        r';\s*delete\s+from',
        r'union\s+select',
        r'exec\s*\(',
        r'<script',
    ]

    search_lower = search_term.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, search_lower):
            logger.warning(f"Blocked potentially malicious search term: {search_term[:50]}")
            return False, "Search term contains invalid characters"

    return True, None


def validate_personal_notes(notes: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate personal notes.

    Args:
        notes: Personal notes to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not notes:
        return True, None  # Notes are optional

    if len(notes) > MAX_NOTES_LENGTH:
        return False, f"Personal notes must be less than {MAX_NOTES_LENGTH} characters"

    return True, None


def validate_condition(condition: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate book condition.

    Args:
        condition: Book condition to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not condition:
        return True, None  # Condition is optional

    valid_conditions = ["New", "Like New", "Very Good", "Good", "Fair", "Poor"]

    if condition not in valid_conditions:
        return False, f"Condition must be one of: {', '.join(valid_conditions)}"

    return True, None


def validate_book_input(
    title: str,
    author: str,
    year: Optional[int] = None,
    isbn: Optional[str] = None,
    publisher: Optional[str] = None,
    condition: Optional[str] = None,
    personal_notes: Optional[str] = None
) -> Tuple[bool, list]:
    """
    Validate all book input fields at once.

    Args:
        title: Book title
        author: Book author
        year: Publication year
        isbn: ISBN
        publisher: Publisher name
        condition: Book condition
        personal_notes: Personal notes

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    # Validate title
    valid, error = validate_title(title)
    if not valid:
        errors.append(f"Title: {error}")

    # Validate author
    valid, error = validate_author(author)
    if not valid:
        errors.append(f"Author: {error}")

    # Validate year
    valid, error = validate_year(year)
    if not valid:
        errors.append(f"Year: {error}")

    # Validate ISBN
    valid, error = validate_isbn(isbn)
    if not valid:
        errors.append(f"ISBN: {error}")

    # Validate publisher
    if publisher:
        publisher = sanitize_string(publisher, MAX_PUBLISHER_LENGTH)
        if len(publisher) > MAX_PUBLISHER_LENGTH:
            errors.append(f"Publisher: Must be less than {MAX_PUBLISHER_LENGTH} characters")

    # Validate condition
    valid, error = validate_condition(condition)
    if not valid:
        errors.append(f"Condition: {error}")

    # Validate personal notes
    valid, error = validate_personal_notes(personal_notes)
    if not valid:
        errors.append(f"Notes: {error}")

    is_valid = len(errors) == 0
    return is_valid, errors


def sanitize_for_display(text: str, max_length: int = 1000) -> str:
    """
    Sanitize text for safe display in UI (prevent XSS if used in web context).

    Args:
        text: Text to sanitize
        max_length: Maximum length for display

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Basic HTML entity encoding for common characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#x27;')

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + '...'

    return text
