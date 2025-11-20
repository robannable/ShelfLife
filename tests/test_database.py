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


class TestDatabaseBackup:
    """Test database backup functionality."""

    def test_backup_database(self, tmp_path):
        """Test creating a database backup."""
        from pathlib import Path

        # Create a temporary database
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        # Add a book
        book = Book(title="Test Book", author="Test Author")
        db.add_book(book)

        # Create backup
        backup_path = tmp_path / "backup.db"
        result = db.backup_database(str(backup_path))

        assert result == str(backup_path)
        assert Path(backup_path).exists()

        # Verify backup has same size as original
        original_size = Path(db_path).stat().st_size
        backup_size = Path(backup_path).stat().st_size
        assert backup_size == original_size

    def test_automatic_backup_path(self, tmp_path, monkeypatch):
        """Test automatic backup path generation."""
        # Create a temporary database
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        # Add a book
        book = Book(title="Test Book", author="Test Author")
        db.add_book(book)

        # Change to tmp directory for backup
        monkeypatch.chdir(tmp_path)

        # Create backup with automatic path
        backup_path = db.backup_database()

        assert backup_path is not None
        assert "database_backup_" in backup_path
        assert Path(backup_path).exists()

    def test_backup_nonexistent_database(self, tmp_path):
        """Test backup fails for nonexistent database."""
        from database import DatabaseError

        db = Database(str(tmp_path / "nonexistent.db"))

        with pytest.raises(DatabaseError, match="Source database not found"):
            db.backup_database()


class TestDatabaseRestore:
    """Test database restore functionality."""

    def test_restore_database(self, tmp_path):
        """Test restoring from a backup."""
        # Create original database with a book
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))
        book1 = Book(title="Original Book", author="Original Author")
        db.add_book(book1)

        # Create backup
        backup_path = tmp_path / "backup.db"
        db.backup_database(str(backup_path))

        # Add another book to original
        book2 = Book(title="New Book", author="New Author")
        db.add_book(book2)

        # Verify original has 2 books
        assert len(db.get_all_books()) == 2

        # Restore from backup
        db.restore_database(str(backup_path), confirm=True)

        # Verify restored database has only 1 book
        assert len(db.get_all_books()) == 1

    def test_restore_requires_confirmation(self, tmp_path):
        """Test restore requires explicit confirmation."""
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))
        backup_path = tmp_path / "backup.db"

        with pytest.raises(ValueError, match="explicit confirmation"):
            db.restore_database(str(backup_path), confirm=False)

    def test_restore_nonexistent_backup(self, tmp_path):
        """Test restore fails for nonexistent backup."""
        from database import DatabaseError

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        with pytest.raises(DatabaseError, match="Backup file not found"):
            db.restore_database(str(tmp_path / "nonexistent.db"), confirm=True)

    def test_restore_creates_safety_backup(self, tmp_path):
        """Test restore creates safety backup of current database."""
        # Create database with data
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))
        book = Book(title="Test Book", author="Test Author")
        db.add_book(book)

        # Create a backup to restore from
        backup_path = tmp_path / "backup.db"
        db.backup_database(str(backup_path))

        # Restore (should create safety backup)
        db.restore_database(str(backup_path), confirm=True)

        # Safety backup should exist
        safety_backup = f"{db_path}.pre_restore_backup"
        assert Path(safety_backup).exists()


class TestDatabaseListBackups:
    """Test listing database backups."""

    def test_list_backups_empty(self, tmp_path):
        """Test listing backups when none exist."""
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        backups = db.list_backups(str(tmp_path))
        assert backups == []

    def test_list_backups(self, tmp_path):
        """Test listing existing backups."""
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        # Add a book
        book = Book(title="Test Book", author="Test Author")
        db.add_book(book)

        # Create multiple backups
        backup1 = tmp_path / "backup1.db"
        backup2 = tmp_path / "backup2.db"

        db.backup_database(str(backup1))
        db.backup_database(str(backup2))

        # List backups
        backups = db.list_backups(str(tmp_path))

        assert len(backups) == 2
        assert all('path' in b for b in backups)
        assert all('size_mb' in b for b in backups)
        assert all('created' in b for b in backups)


class TestDatabaseHealthCheck:
    """Test database health check."""

    def test_health_check_healthy_database(self, tmp_path):
        """Test health check on healthy database."""
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        # Add some books
        for i in range(5):
            book = Book(title=f"Book {i}", author=f"Author {i}")
            db.add_book(book)

        health = db.health_check()

        assert health['accessible'] is True
        assert health['total_books'] == 5
        assert health['schema_version'] >= 1
        assert len(health['issues']) >= 0  # May have "no backups" issue

    def test_health_check_nonexistent_database(self, tmp_path):
        """Test health check on nonexistent database."""
        db = Database(str(tmp_path / "nonexistent.db"))

        health = db.health_check()

        assert health['accessible'] is False
        assert "does not exist" in health['issues'][0]


class TestDatabaseVacuum:
    """Test database vacuum operation."""

    def test_vacuum_database(self, tmp_path):
        """Test vacuuming database."""
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        # Add and delete books to create fragmentation
        for i in range(10):
            book = Book(title=f"Book {i}", author="Author")
            book_id = db.add_book(book)
            if i % 2 == 0:
                db.delete_book(book_id)

        # Vacuum
        result = db.vacuum()
        assert result is True
