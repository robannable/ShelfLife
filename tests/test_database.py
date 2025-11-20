"""
Tests for database module.
"""
import pytest
import sqlite3
from datetime import datetime
from database import Database, DatabaseError
from models import Book, BookMetadata


@pytest.fixture
def in_memory_db():
    """Create an in-memory database for testing."""
    db = Database(":memory:")
    yield db
    # Cleanup happens automatically with in-memory DB


@pytest.fixture
def sample_book():
    """Create a sample book for testing."""
    metadata = BookMetadata(
        synopsis="A dystopian novel",
        genre=["Fiction", "Dystopian"],
        themes=["totalitarianism", "surveillance"],
        year=1949
    )
    return Book(
        title="1984",
        author="George Orwell",
        year=1949,
        isbn="0451524934",
        condition="Good",
        metadata=metadata
    )


class TestDatabaseInitialization:
    """Test database initialization."""

    def test_create_database(self, in_memory_db):
        """Test database is created successfully."""
        assert in_memory_db is not None

    def test_schema_creation(self, in_memory_db):
        """Test schema tables are created."""
        with in_memory_db.get_connection() as conn:
            cursor = conn.cursor()
            # Check if books table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='books'"
            )
            result = cursor.fetchone()
            assert result is not None

    def test_schema_version_tracking(self, in_memory_db):
        """Test schema version is tracked."""
        with in_memory_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(version) FROM schema_version")
            version = cursor.fetchone()[0]
            assert version >= 1


class TestAddBook:
    """Test adding books to database."""

    def test_add_valid_book(self, in_memory_db, sample_book):
        """Test adding a valid book."""
        book_id = in_memory_db.add_book(sample_book)
        assert book_id is not None
        assert book_id > 0

    def test_add_book_without_metadata(self, in_memory_db):
        """Test adding a book without metadata."""
        book = Book(title="Test Book", author="Test Author")
        book_id = in_memory_db.add_book(book)
        assert book_id > 0

    def test_add_book_with_personal_notes(self, in_memory_db):
        """Test adding a book with personal notes."""
        book = Book(
            title="Test Book",
            author="Test Author",
            personal_notes="Great read!"
        )
        book_id = in_memory_db.add_book(book)
        assert book_id > 0

        # Verify notes were saved
        retrieved = in_memory_db.get_book(book_id)
        # Personal notes should be at index 11 based on schema
        if len(retrieved) > 11:
            assert retrieved[11] == "Great read!"


class TestGetBook:
    """Test retrieving books from database."""

    def test_get_existing_book(self, in_memory_db, sample_book):
        """Test getting an existing book."""
        book_id = in_memory_db.add_book(sample_book)
        retrieved = in_memory_db.get_book(book_id)

        assert retrieved is not None
        assert retrieved[0] == book_id
        assert retrieved[1] == "1984"
        assert retrieved[2] == "George Orwell"

    def test_get_nonexistent_book(self, in_memory_db):
        """Test getting a book that doesn't exist."""
        retrieved = in_memory_db.get_book(9999)
        assert retrieved is None


class TestUpdateBook:
    """Test updating books in database."""

    def test_update_existing_book(self, in_memory_db, sample_book):
        """Test updating an existing book."""
        book_id = in_memory_db.add_book(sample_book)

        # Update the book
        sample_book.title = "Nineteen Eighty-Four"
        sample_book.year = 1950
        success = in_memory_db.update_book(book_id, sample_book)

        assert success is True

        # Verify update
        retrieved = in_memory_db.get_book(book_id)
        assert retrieved[1] == "Nineteen Eighty-Four"
        assert retrieved[3] == 1950

    def test_update_nonexistent_book(self, in_memory_db, sample_book):
        """Test updating a book that doesn't exist."""
        success = in_memory_db.update_book(9999, sample_book)
        assert success is False


class TestDeleteBook:
    """Test deleting books from database."""

    def test_delete_existing_book(self, in_memory_db, sample_book):
        """Test deleting an existing book."""
        book_id = in_memory_db.add_book(sample_book)
        success = in_memory_db.delete_book(book_id)

        assert success is True

        # Verify deletion
        retrieved = in_memory_db.get_book(book_id)
        assert retrieved is None

    def test_delete_nonexistent_book(self, in_memory_db):
        """Test deleting a book that doesn't exist."""
        success = in_memory_db.delete_book(9999)
        assert success is False


class TestSearchBooks:
    """Test searching for books."""

    def test_search_all_books(self, in_memory_db, sample_book):
        """Test searching without filter returns all books."""
        in_memory_db.add_book(sample_book)

        book2 = Book(title="Animal Farm", author="George Orwell")
        in_memory_db.add_book(book2)

        results = in_memory_db.search_books()
        assert len(results) == 2

    def test_search_by_title(self, in_memory_db, sample_book):
        """Test searching by title."""
        in_memory_db.add_book(sample_book)

        book2 = Book(title="Animal Farm", author="George Orwell")
        in_memory_db.add_book(book2)

        results = in_memory_db.search_books("1984")
        assert len(results) == 1
        assert results[0][1] == "1984"

    def test_search_by_author(self, in_memory_db, sample_book):
        """Test searching by author."""
        in_memory_db.add_book(sample_book)

        book2 = Book(title="To Kill a Mockingbird", author="Harper Lee")
        in_memory_db.add_book(book2)

        results = in_memory_db.search_books("Orwell")
        assert len(results) == 1
        assert results[0][2] == "George Orwell"

    def test_search_case_insensitive(self, in_memory_db, sample_book):
        """Test search is case insensitive."""
        in_memory_db.add_book(sample_book)

        results = in_memory_db.search_books("orwell")
        assert len(results) == 1

    def test_sort_by_title(self, in_memory_db):
        """Test sorting by title."""
        book1 = Book(title="Zebra", author="Author A")
        book2 = Book(title="Apple", author="Author B")

        in_memory_db.add_book(book1)
        in_memory_db.add_book(book2)

        results = in_memory_db.search_books(sort_by="Title")
        assert results[0][1] == "Apple"
        assert results[1][1] == "Zebra"

    def test_sort_by_author(self, in_memory_db):
        """Test sorting by author."""
        book1 = Book(title="Book 1", author="Zoe Smith")
        book2 = Book(title="Book 2", author="Alice Johnson")

        in_memory_db.add_book(book1)
        in_memory_db.add_book(book2)

        results = in_memory_db.search_books(sort_by="Author")
        assert results[0][2] == "Alice Johnson"
        assert results[1][2] == "Zoe Smith"

    def test_invalid_sort_column(self, in_memory_db, sample_book):
        """Test invalid sort column defaults to title."""
        in_memory_db.add_book(sample_book)

        # Should not raise error, should default to title
        results = in_memory_db.search_books(sort_by="InvalidColumn")
        assert len(results) == 1


class TestGetAllBooks:
    """Test getting all books."""

    def test_get_all_books_empty(self, in_memory_db):
        """Test getting all books from empty database."""
        results = in_memory_db.get_all_books()
        assert len(results) == 0

    def test_get_all_books_multiple(self, in_memory_db):
        """Test getting all books when multiple exist."""
        for i in range(5):
            book = Book(title=f"Book {i}", author=f"Author {i}")
            in_memory_db.add_book(book)

        results = in_memory_db.get_all_books()
        assert len(results) == 5


class TestGetBooksWithMetadata:
    """Test getting books with metadata."""

    def test_get_books_with_metadata(self, in_memory_db, sample_book):
        """Test getting only books that have metadata."""
        # Add book with metadata
        in_memory_db.add_book(sample_book)

        # Add book without metadata
        book_no_meta = Book(title="No Metadata", author="Author")
        in_memory_db.add_book(book_no_meta)

        results = in_memory_db.get_books_with_metadata()
        assert len(results) == 1
        assert results[0][1] == "1984"


class TestUpdateMetadata:
    """Test updating book metadata."""

    def test_update_metadata(self, in_memory_db, sample_book):
        """Test updating metadata for a book."""
        book_id = in_memory_db.add_book(sample_book)

        new_metadata = BookMetadata(
            synopsis="Updated synopsis",
            genre=["Fiction"],
            themes=["new theme"]
        )

        success = in_memory_db.update_metadata(book_id, new_metadata)
        assert success is True

        # Verify update
        retrieved = in_memory_db.get_book(book_id)
        assert "Updated synopsis" in retrieved[8]


class TestGetStatistics:
    """Test getting database statistics."""

    def test_statistics_empty_db(self, in_memory_db):
        """Test statistics on empty database."""
        stats = in_memory_db.get_statistics()

        assert stats['total_books'] == 0
        assert stats['unique_authors'] == 0
        assert stats['avg_pub_year'] is None

    def test_statistics_with_books(self, in_memory_db):
        """Test statistics with books."""
        book1 = Book(title="Book 1", author="Author A", year=2000)
        book2 = Book(title="Book 2", author="Author A", year=2010)
        book3 = Book(title="Book 3", author="Author B", year=2020)

        in_memory_db.add_book(book1)
        in_memory_db.add_book(book2)
        in_memory_db.add_book(book3)

        stats = in_memory_db.get_statistics()

        assert stats['total_books'] == 3
        assert stats['unique_authors'] == 2
        assert stats['avg_pub_year'] == 2010


class TestTransactionRollback:
    """Test transaction rollback on errors."""

    def test_rollback_on_error(self, in_memory_db):
        """Test that database rolls back on error."""
        # This test verifies the context manager's rollback behavior
        initial_count = len(in_memory_db.get_all_books())

        try:
            with in_memory_db.get_connection() as conn:
                cursor = conn.cursor()
                # Insert a book
                cursor.execute(
                    "INSERT INTO books (title, author) VALUES (?, ?)",
                    ("Test", "Author")
                )
                # Cause an error
                raise Exception("Simulated error")
        except DatabaseError:
            pass

        # Verify no book was added due to rollback
        final_count = len(in_memory_db.get_all_books())
        assert final_count == initial_count
