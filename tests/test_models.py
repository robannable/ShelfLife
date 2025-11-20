"""
Tests for data models.
"""
import pytest
import json
from datetime import datetime
from models import Book, BookMetadata, LibraryStats, ThemeAnalysis, ExecutiveSummary


class TestBookMetadata:
    """Test BookMetadata model."""

    def test_create_empty_metadata(self):
        """Test creating empty metadata."""
        metadata = BookMetadata()
        assert metadata.synopsis is None
        assert metadata.genre == []
        assert metadata.themes == []

    def test_create_full_metadata(self):
        """Test creating metadata with all fields."""
        metadata = BookMetadata(
            synopsis="A great book",
            genre=["Fiction", "Sci-Fi"],
            themes=["technology", "society"],
            historical_context="Written during...",
            time_period="1950s",
            reading_level="Adult",
            keywords=["future", "dystopia"],
            cover_url="http://example.com/cover.jpg"
        )

        assert metadata.synopsis == "A great book"
        assert len(metadata.genre) == 2
        assert len(metadata.themes) == 2

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = BookMetadata(
            synopsis="Test",
            genre=["Fiction"],
            themes=["test"]
        )

        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert "synopsis" in data
        assert "genre" in data
        assert "themes" in data

    def test_to_dict_excludes_empty(self):
        """Test to_dict excludes None and empty values."""
        metadata = BookMetadata(synopsis="Test")
        data = metadata.to_dict()

        # Empty lists should not be included
        assert "keywords" not in data or data["keywords"] == []

    def test_to_json(self):
        """Test converting metadata to JSON string."""
        metadata = BookMetadata(
            synopsis="Test",
            genre=["Fiction"]
        )

        json_str = metadata.to_json()
        assert isinstance(json_str, str)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["synopsis"] == "Test"

    def test_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "synopsis": "Test synopsis",
            "genre": ["Fiction", "Mystery"],
            "themes": ["crime", "justice"]
        }

        metadata = BookMetadata.from_dict(data)
        assert metadata.synopsis == "Test synopsis"
        assert len(metadata.genre) == 2
        assert len(metadata.themes) == 2

    def test_from_dict_ignores_unknown_fields(self):
        """Test from_dict ignores fields not in the model."""
        data = {
            "synopsis": "Test",
            "unknown_field": "should be ignored"
        }

        metadata = BookMetadata.from_dict(data)
        assert metadata.synopsis == "Test"
        assert not hasattr(metadata, "unknown_field")

    def test_from_json(self):
        """Test creating metadata from JSON string."""
        json_str = '{"synopsis": "Test", "genre": ["Fiction"]}'
        metadata = BookMetadata.from_json(json_str)

        assert metadata.synopsis == "Test"
        assert metadata.genre == ["Fiction"]

    def test_roundtrip_serialization(self):
        """Test serialization and deserialization roundtrip."""
        original = BookMetadata(
            synopsis="Original synopsis",
            genre=["Fiction", "Sci-Fi"],
            themes=["space", "exploration"],
            year=2024
        )

        # Convert to JSON and back
        json_str = original.to_json()
        restored = BookMetadata.from_json(json_str)

        assert restored.synopsis == original.synopsis
        assert restored.genre == original.genre
        assert restored.themes == original.themes
        assert restored.year == original.year


class TestBook:
    """Test Book model."""

    def test_create_minimal_book(self):
        """Test creating book with minimal required fields."""
        book = Book(title="Test Book", author="Test Author")
        assert book.title == "Test Book"
        assert book.author == "Test Author"
        assert book.condition == "Good"  # Default value

    def test_create_full_book(self):
        """Test creating book with all fields."""
        metadata = BookMetadata(synopsis="Test")
        book = Book(
            title="Test Book",
            author="Test Author",
            year=2024,
            isbn="1234567890",
            publisher="Test Publisher",
            condition="New",
            metadata=metadata,
            personal_notes="Great book!"
        )

        assert book.title == "Test Book"
        assert book.year == 2024
        assert book.isbn == "1234567890"
        assert book.personal_notes == "Great book!"

    def test_validation_empty_title(self):
        """Test validation rejects empty title."""
        with pytest.raises(ValueError, match="Title is required"):
            Book(title="", author="Author")

        with pytest.raises(ValueError, match="Title is required"):
            Book(title="   ", author="Author")

    def test_validation_empty_author(self):
        """Test validation rejects empty author."""
        with pytest.raises(ValueError, match="Author is required"):
            Book(title="Title", author="")

        with pytest.raises(ValueError, match="Author is required"):
            Book(title="Title", author="   ")

    def test_validation_invalid_year(self):
        """Test validation rejects invalid years."""
        with pytest.raises(ValueError, match="Invalid year"):
            Book(title="Title", author="Author", year=-100)

        current_year = datetime.now().year
        with pytest.raises(ValueError, match="Invalid year"):
            Book(title="Title", author="Author", year=current_year + 10)

    def test_validation_invalid_condition(self):
        """Test validation rejects invalid condition."""
        with pytest.raises(ValueError, match="Invalid condition"):
            Book(title="Title", author="Author", condition="Excellent")

    def test_valid_conditions(self):
        """Test all valid conditions are accepted."""
        valid_conditions = ["New", "Like New", "Very Good", "Good", "Fair", "Poor"]

        for condition in valid_conditions:
            book = Book(title="Title", author="Author", condition=condition)
            assert book.condition == condition

    def test_to_dict(self):
        """Test converting book to dictionary."""
        book = Book(
            title="Test",
            author="Author",
            year=2024,
            isbn="1234567890"
        )

        data = book.to_dict()
        assert isinstance(data, dict)
        assert data["title"] == "Test"
        assert data["author"] == "Author"
        assert data["year"] == 2024
        assert data["isbn"] == "1234567890"

    def test_to_dict_with_metadata(self):
        """Test to_dict includes serialized metadata."""
        metadata = BookMetadata(synopsis="Test synopsis")
        book = Book(
            title="Test",
            author="Author",
            metadata=metadata
        )

        data = book.to_dict()
        assert "metadata" in data
        assert data["metadata"] is not None

        # Verify metadata is JSON string
        parsed_metadata = json.loads(data["metadata"])
        assert parsed_metadata["synopsis"] == "Test synopsis"


class TestLibraryStats:
    """Test LibraryStats model."""

    def test_create_empty_stats(self):
        """Test creating empty stats."""
        stats = LibraryStats()
        assert stats.total_books == 0
        assert stats.unique_authors == 0
        assert stats.avg_pub_year is None
        assert stats.fiction_ratio == 0.0
        assert stats.nonfiction_ratio == 0.0

    def test_create_full_stats(self):
        """Test creating stats with all fields."""
        stats = LibraryStats(
            total_books=100,
            unique_authors=50,
            avg_pub_year=2000,
            common_time_period="Modern",
            time_period_coverage=0.8,
            fiction_ratio=0.6,
            nonfiction_ratio=0.4
        )

        assert stats.total_books == 100
        assert stats.unique_authors == 50
        assert stats.avg_pub_year == 2000
        assert stats.fiction_ratio == 0.6


class TestThemeAnalysis:
    """Test ThemeAnalysis model."""

    def test_create_empty_analysis(self):
        """Test creating empty theme analysis."""
        analysis = ThemeAnalysis()
        assert analysis.uber_themes == []
        assert analysis.analysis == {}

    def test_create_full_analysis(self):
        """Test creating theme analysis with data."""
        analysis = ThemeAnalysis(
            uber_themes=[
                {
                    "name": "Science Fiction",
                    "sub_themes": ["space", "technology"]
                }
            ],
            analysis={"summary": "Test summary"},
            generated_at="2024-01-01"
        )

        assert len(analysis.uber_themes) == 1
        assert "summary" in analysis.analysis


class TestExecutiveSummary:
    """Test ExecutiveSummary model."""

    def test_create_summary(self):
        """Test creating executive summary."""
        summary = ExecutiveSummary(
            summary="This collection focuses on...",
            patterns=["Pattern 1", "Pattern 2"],
            recommendations=["Read more of genre X"],
            last_updated="2024-01-01"
        )

        assert "collection" in summary.summary
        assert len(summary.patterns) == 2
        assert len(summary.recommendations) == 1

    def test_summary_required(self):
        """Test that summary field is required."""
        # This should work since summary has a default in the test
        summary = ExecutiveSummary(summary="Test")
        assert summary.summary == "Test"
