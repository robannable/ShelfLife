"""
Tests for validation module.
"""
import pytest
from validation import (
    validate_title,
    validate_author,
    validate_year,
    validate_isbn,
    validate_search_term,
    validate_personal_notes,
    validate_condition,
    validate_book_input,
    sanitize_string,
    sanitize_for_display,
    ValidationError
)


class TestValidateTitle:
    """Test title validation."""

    def test_valid_title(self):
        """Test valid titles."""
        valid, error = validate_title("1984")
        assert valid is True
        assert error is None

        valid, error = validate_title("The Lord of the Rings")
        assert valid is True
        assert error is None

    def test_empty_title(self):
        """Test empty title is rejected."""
        valid, error = validate_title("")
        assert valid is False
        assert "required" in error.lower()

        valid, error = validate_title("   ")
        assert valid is False

    def test_very_long_title(self):
        """Test very long titles are handled."""
        long_title = "A" * 600
        valid, error = validate_title(long_title)
        assert valid is False
        assert "500" in error

    def test_title_with_special_chars(self):
        """Test titles with special characters."""
        valid, error = validate_title("It's a Wonderful Life!")
        assert valid is True

        valid, error = validate_title("Book: The Sequel")
        assert valid is True


class TestValidateAuthor:
    """Test author validation."""

    def test_valid_author(self):
        """Test valid author names."""
        valid, error = validate_author("George Orwell")
        assert valid is True
        assert error is None

    def test_empty_author(self):
        """Test empty author is rejected."""
        valid, error = validate_author("")
        assert valid is False
        assert "required" in error.lower()

    def test_author_with_special_chars(self):
        """Test author names with special characters."""
        valid, error = validate_author("Jean-Paul Sartre")
        assert valid is True

        valid, error = validate_author("O'Brien")
        assert valid is True


class TestValidateYear:
    """Test year validation."""

    def test_valid_year(self):
        """Test valid years."""
        valid, error = validate_year(1984)
        assert valid is True
        assert error is None

        valid, error = validate_year(2024)
        assert valid is True

    def test_none_year(self):
        """Test None year is acceptable."""
        valid, error = validate_year(None)
        assert valid is True

    def test_future_year(self):
        """Test future years are handled."""
        valid, error = validate_year(2050)
        assert valid is False
        assert "future" in error.lower()

    def test_ancient_year(self):
        """Test ancient years."""
        valid, error = validate_year(-5000)
        assert valid is False

        valid, error = validate_year(-1000)
        assert valid is True


class TestValidateISBN:
    """Test ISBN validation."""

    def test_valid_isbn(self):
        """Test valid ISBNs."""
        valid, error = validate_isbn("1234567890")
        assert valid is True

        valid, error = validate_isbn("978-1234567890")
        assert valid is True

        valid, error = validate_isbn("123456789X")
        assert valid is True

    def test_none_isbn(self):
        """Test None ISBN is acceptable."""
        valid, error = validate_isbn(None)
        assert valid is True

        valid, error = validate_isbn("")
        assert valid is True

    def test_invalid_isbn_length(self):
        """Test ISBN with invalid length."""
        valid, error = validate_isbn("12345")
        assert valid is False
        assert "10 or 13" in error

    def test_invalid_isbn_chars(self):
        """Test ISBN with invalid characters."""
        valid, error = validate_isbn("123ABC7890")
        assert valid is False


class TestValidateSearchTerm:
    """Test search term validation."""

    def test_valid_search(self):
        """Test valid search terms."""
        valid, error = validate_search_term("orwell")
        assert valid is True

        valid, error = validate_search_term("1984")
        assert valid is True

    def test_empty_search(self):
        """Test empty search is acceptable."""
        valid, error = validate_search_term("")
        assert valid is True

    def test_sql_injection_attempt(self):
        """Test SQL injection patterns are blocked."""
        valid, error = validate_search_term("'; DROP TABLE books;--")
        assert valid is False

        valid, error = validate_search_term("UNION SELECT * FROM users")
        assert valid is False

    def test_xss_attempt(self):
        """Test XSS patterns are blocked."""
        valid, error = validate_search_term("<script>alert('xss')</script>")
        assert valid is False


class TestValidateCondition:
    """Test condition validation."""

    def test_valid_condition(self):
        """Test valid conditions."""
        valid_conditions = ["New", "Like New", "Very Good", "Good", "Fair", "Poor"]
        for condition in valid_conditions:
            valid, error = validate_condition(condition)
            assert valid is True
            assert error is None

    def test_invalid_condition(self):
        """Test invalid condition."""
        valid, error = validate_condition("Excellent")
        assert valid is False

    def test_none_condition(self):
        """Test None condition is acceptable."""
        valid, error = validate_condition(None)
        assert valid is True


class TestValidatePersonalNotes:
    """Test personal notes validation."""

    def test_valid_notes(self):
        """Test valid notes."""
        valid, error = validate_personal_notes("This is a great book!")
        assert valid is True

    def test_none_notes(self):
        """Test None notes are acceptable."""
        valid, error = validate_personal_notes(None)
        assert valid is True

    def test_very_long_notes(self):
        """Test very long notes are rejected."""
        long_notes = "A" * 20000
        valid, error = validate_personal_notes(long_notes)
        assert valid is False
        assert "10000" in error


class TestValidateBookInput:
    """Test comprehensive book input validation."""

    def test_valid_book_input(self):
        """Test valid book input."""
        is_valid, errors = validate_book_input(
            title="1984",
            author="George Orwell",
            year=1949,
            isbn="0451524934",
            condition="Good"
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_book_input(self):
        """Test invalid book input."""
        is_valid, errors = validate_book_input(
            title="",
            author="",
            year=2050,
            isbn="invalid",
            condition="Excellent"
        )
        assert is_valid is False
        assert len(errors) > 0

    def test_minimal_valid_input(self):
        """Test minimal valid input (only required fields)."""
        is_valid, errors = validate_book_input(
            title="Book Title",
            author="Author Name"
        )
        assert is_valid is True
        assert len(errors) == 0


class TestSanitizeString:
    """Test string sanitization."""

    def test_trim_whitespace(self):
        """Test whitespace trimming."""
        result = sanitize_string("  hello  ", 100)
        assert result == "hello"

    def test_normalize_spaces(self):
        """Test multiple spaces are normalized."""
        result = sanitize_string("hello    world", 100)
        assert result == "hello world"

    def test_truncate_long_string(self):
        """Test long strings are truncated."""
        result = sanitize_string("A" * 200, 50)
        assert len(result) == 50


class TestSanitizeForDisplay:
    """Test display sanitization."""

    def test_html_entity_encoding(self):
        """Test HTML entities are encoded."""
        result = sanitize_for_display("<script>alert('xss')</script>")
        assert "&lt;" in result
        assert "&gt;" in result
        assert "<script>" not in result

    def test_truncate_for_display(self):
        """Test truncation for display."""
        result = sanitize_for_display("A" * 2000, max_length=100)
        assert len(result) <= 103  # 100 + "..."
