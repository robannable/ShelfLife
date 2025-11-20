"""
Database operations for ShelfLife application.
"""
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from models import Book, BookMetadata
from logger import get_logger

logger = get_logger(__name__)


class DatabaseError(Exception):
    """Custom exception for database errors."""
    pass


class Database:
    """Manages database operations for the book collection."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection."""
        if db_path is None:
            data_dir = Path('data')
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / 'database.db')

        self.db_path = db_path
        logger.info(f"Database initialized at: {db_path}")
        self._init_schema()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {str(e)}", exc_info=True)
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema with versioning."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()

                # Create version table
                c.execute('''
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Get current version
                current_version = c.execute(
                    'SELECT MAX(version) FROM schema_version'
                ).fetchone()[0] or 0

                # Define schema updates
                schema_updates = {
                    1: '''
                        CREATE TABLE IF NOT EXISTS books (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            title TEXT NOT NULL,
                            author TEXT NOT NULL,
                            year INTEGER,
                            isbn TEXT,
                            publisher TEXT,
                            condition TEXT,
                            cover_image BLOB,
                            metadata JSON,
                            created_at TIMESTAMP,
                            updated_at TIMESTAMP
                        );

                        CREATE INDEX IF NOT EXISTS idx_title ON books(title);
                        CREATE INDEX IF NOT EXISTS idx_author ON books(author);
                        CREATE INDEX IF NOT EXISTS idx_year ON books(year);
                    ''',
                    2: '''
                        ALTER TABLE books ADD COLUMN personal_notes TEXT;
                    '''
                }

                # Apply updates
                for version, update_sql in schema_updates.items():
                    if version > current_version:
                        try:
                            c.executescript(update_sql)
                            c.execute('INSERT INTO schema_version (version) VALUES (?)', (version,))
                            logger.info(f"Applied database schema update version {version}")
                        except sqlite3.OperationalError as e:
                            logger.warning(f"Schema update {version} may already be applied: {str(e)}")
                            continue

                conn.commit()
                logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database schema: {str(e)}", exc_info=True)
            raise DatabaseError(f"Schema initialization failed: {str(e)}")

    def add_book(self, book: Book) -> int:
        """Add a new book to the database."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                book.created_at = datetime.now()

                c.execute('''
                    INSERT INTO books (
                        title, author, year, isbn, publisher,
                        condition, cover_image, metadata, personal_notes, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    book.title, book.author, book.year, book.isbn, book.publisher,
                    book.condition, book.cover_image,
                    book.metadata.to_json() if book.metadata else None,
                    book.personal_notes, book.created_at.isoformat()
                ))

                book_id = c.lastrowid
                logger.info(f"Added book: {book.title} by {book.author} (ID: {book_id})")
                return book_id

        except Exception as e:
            logger.error(f"Failed to add book '{book.title}': {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to add book: {str(e)}")

    def update_book(self, book_id: int, book: Book) -> bool:
        """Update an existing book."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                book.updated_at = datetime.now()

                c.execute('''
                    UPDATE books
                    SET title = ?, author = ?, year = ?, isbn = ?,
                        publisher = ?, condition = ?, cover_image = ?,
                        metadata = ?, personal_notes = ?, updated_at = ?
                    WHERE id = ?
                ''', (
                    book.title, book.author, book.year, book.isbn,
                    book.publisher, book.condition, book.cover_image,
                    book.metadata.to_json() if book.metadata else None,
                    book.personal_notes, book.updated_at.isoformat(), book_id
                ))

                success = c.rowcount > 0
                if success:
                    logger.info(f"Updated book ID {book_id}: {book.title}")
                else:
                    logger.warning(f"Book ID {book_id} not found for update")

                return success

        except Exception as e:
            logger.error(f"Failed to update book ID {book_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to update book: {str(e)}")

    def delete_book(self, book_id: int) -> bool:
        """Delete a book from the database."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('DELETE FROM books WHERE id = ?', (book_id,))

                success = c.rowcount > 0
                if success:
                    logger.info(f"Deleted book ID {book_id}")
                else:
                    logger.warning(f"Book ID {book_id} not found for deletion")

                return success

        except Exception as e:
            logger.error(f"Failed to delete book ID {book_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to delete book: {str(e)}")

    def get_book(self, book_id: int) -> Optional[Tuple]:
        """Get a book by ID."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                result = c.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()

                if result:
                    logger.debug(f"Retrieved book ID {book_id}")
                else:
                    logger.debug(f"Book ID {book_id} not found")

                return result

        except Exception as e:
            logger.error(f"Failed to get book ID {book_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve book: {str(e)}")

    def search_books(self, search_term: str = "", sort_by: str = "title") -> List[Tuple]:
        """Search books with optional filtering and sorting."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()

                query = 'SELECT * FROM books'
                params = []

                if search_term:
                    query += ' WHERE title LIKE ? OR author LIKE ?'
                    params.extend([f'%{search_term}%', f'%{search_term}%'])

                # Whitelist of allowed sort columns to prevent SQL injection
                ALLOWED_SORT_COLUMNS = {
                    "Title": "title",
                    "Author": "author",
                    "Year": "year",
                    "Recent": "created_at"
                }

                # Validate and sanitize sort column
                sort_column = ALLOWED_SORT_COLUMNS.get(sort_by)
                if not sort_column:
                    logger.warning(f"Invalid sort_by value: {sort_by}. Defaulting to 'title'")
                    sort_column = "title"

                if sort_column == "created_at":
                    query += ' ORDER BY created_at DESC'
                else:
                    query += f' ORDER BY {sort_column}'

                results = c.execute(query, params).fetchall()
                logger.debug(f"Search returned {len(results)} books")
                return results

        except sqlite3.Error as e:
            logger.error(f"Database error while searching books: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to search books: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while searching books: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to search books: {str(e)}")

    def get_all_books(self) -> List[Tuple]:
        """Get all books from the database."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                results = c.execute('SELECT * FROM books').fetchall()
                logger.debug(f"Retrieved {len(results)} books")
                return results

        except Exception as e:
            logger.error(f"Failed to get all books: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve books: {str(e)}")

    def get_books_with_metadata(self) -> List[Tuple]:
        """Get all books that have metadata."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                results = c.execute(
                    'SELECT * FROM books WHERE metadata IS NOT NULL'
                ).fetchall()
                logger.debug(f"Retrieved {len(results)} books with metadata")
                return results

        except Exception as e:
            logger.error(f"Failed to get books with metadata: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve books: {str(e)}")

    def update_metadata(self, book_id: int, metadata: BookMetadata) -> bool:
        """Update only the metadata for a book."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''
                    UPDATE books
                    SET metadata = ?, updated_at = ?
                    WHERE id = ?
                ''', (metadata.to_json(), datetime.now().isoformat(), book_id))

                success = c.rowcount > 0
                if success:
                    logger.info(f"Updated metadata for book ID {book_id}")
                else:
                    logger.warning(f"Book ID {book_id} not found for metadata update")

                return success

        except Exception as e:
            logger.error(f"Failed to update metadata for book ID {book_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to update metadata: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the collection."""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()

                stats = {}
                stats['total_books'] = c.execute('SELECT COUNT(*) FROM books').fetchone()[0]
                stats['unique_authors'] = c.execute(
                    'SELECT COUNT(DISTINCT author) FROM books'
                ).fetchone()[0]

                avg_year = c.execute('''
                    SELECT AVG(CAST(year AS FLOAT))
                    FROM books WHERE year IS NOT NULL
                ''').fetchone()[0]

                stats['avg_pub_year'] = round(avg_year) if avg_year else None

                logger.debug(f"Retrieved statistics: {stats}")
                return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve statistics: {str(e)}")

    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the database.

        Args:
            backup_path: Path for backup file. If None, creates timestamped backup

        Returns:
            Path to backup file

        Raises:
            DatabaseError: If backup fails
        """
        import shutil

        try:
            if backup_path is None:
                # Create backups directory
                backup_dir = Path("data/backups")
                backup_dir.mkdir(parents=True, exist_ok=True)

                # Generate timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = str(backup_dir / f"database_backup_{timestamp}.db")

            # Ensure source database exists
            if not Path(self.db_path).exists():
                raise DatabaseError(f"Source database not found: {self.db_path}")

            # Create backup
            shutil.copy2(self.db_path, backup_path)

            # Verify backup
            backup_size = Path(backup_path).stat().st_size
            original_size = Path(self.db_path).stat().st_size

            if backup_size != original_size:
                logger.warning(
                    f"Backup size ({backup_size}) differs from original ({original_size})"
                )

            logger.info(f"Database backed up successfully to: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to backup database: {str(e)}", exc_info=True)
            raise DatabaseError(f"Backup failed: {str(e)}")

    def restore_database(self, backup_path: str, confirm: bool = False) -> bool:
        """
        Restore database from a backup file.

        Args:
            backup_path: Path to backup file
            confirm: Must be True to actually restore (safety check)

        Returns:
            True if restore successful

        Raises:
            DatabaseError: If restore fails
            ValueError: If confirm is not True
        """
        import shutil

        if not confirm:
            raise ValueError(
                "Restore operation requires explicit confirmation. "
                "Set confirm=True to proceed."
            )

        try:
            # Verify backup file exists
            if not Path(backup_path).exists():
                raise DatabaseError(f"Backup file not found: {backup_path}")

            # Create backup of current database before restoring
            current_backup = None
            if Path(self.db_path).exists():
                current_backup = self.backup_database(
                    f"{self.db_path}.pre_restore_backup"
                )
                logger.info(f"Created safety backup: {current_backup}")

            # Restore from backup
            shutil.copy2(backup_path, self.db_path)

            # Verify restored database
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Try a simple query to verify database integrity
                cursor.execute("SELECT COUNT(*) FROM books")
                count = cursor.fetchone()[0]
                logger.info(f"Database restored successfully. Contains {count} books.")

            logger.info(f"Database restored from: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore database: {str(e)}", exc_info=True)

            # Attempt to restore from safety backup if it exists
            if current_backup and Path(current_backup).exists():
                logger.warning("Attempting to restore from safety backup...")
                try:
                    shutil.copy2(current_backup, self.db_path)
                    logger.info("Restored from safety backup")
                except Exception as restore_error:
                    logger.critical(
                        f"Failed to restore from safety backup: {str(restore_error)}"
                    )

            raise DatabaseError(f"Restore failed: {str(e)}")

    def list_backups(self, backup_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available database backups.

        Args:
            backup_dir: Directory to search for backups. Defaults to data/backups

        Returns:
            List of dictionaries with backup information
        """
        try:
            if backup_dir is None:
                backup_dir = Path("data/backups")
            else:
                backup_dir = Path(backup_dir)

            if not backup_dir.exists():
                logger.debug(f"Backup directory does not exist: {backup_dir}")
                return []

            backups = []
            for backup_file in backup_dir.glob("*.db"):
                stat = backup_file.stat()
                backups.append({
                    'path': str(backup_file),
                    'name': backup_file.name,
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'created': datetime.fromtimestamp(stat.st_ctime),
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })

            # Sort by creation time, newest first
            backups.sort(key=lambda x: x['created'], reverse=True)

            logger.debug(f"Found {len(backups)} backup(s)")
            return backups

        except Exception as e:
            logger.error(f"Error listing backups: {str(e)}", exc_info=True)
            return []

    def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.

        Returns:
            Dictionary with health check results
        """
        health = {
            'accessible': False,
            'size_mb': 0,
            'total_books': 0,
            'schema_version': 0,
            'last_backup': None,
            'issues': []
        }

        try:
            # Check if database file exists and is accessible
            db_path = Path(self.db_path)
            if not db_path.exists():
                health['issues'].append("Database file does not exist")
                return health

            health['size_mb'] = round(db_path.stat().st_size / (1024 * 1024), 2)

            # Check database connectivity and integrity
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Check schema version
                try:
                    cursor.execute("SELECT MAX(version) FROM schema_version")
                    health['schema_version'] = cursor.fetchone()[0] or 0
                except sqlite3.OperationalError:
                    health['issues'].append("schema_version table missing")

                # Check books table
                try:
                    cursor.execute("SELECT COUNT(*) FROM books")
                    health['total_books'] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    health['issues'].append("books table missing or inaccessible")

                # Run integrity check
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                if integrity_result != "ok":
                    health['issues'].append(f"Integrity check failed: {integrity_result}")

            health['accessible'] = True

            # Check for recent backups
            backups = self.list_backups()
            if backups:
                health['last_backup'] = backups[0]['created']
            else:
                health['issues'].append("No backups found")

            logger.info(f"Health check completed: {len(health['issues'])} issues found")

        except Exception as e:
            health['issues'].append(f"Health check error: {str(e)}")
            logger.error(f"Health check failed: {str(e)}", exc_info=True)

        return health

    def vacuum(self) -> bool:
        """
        Vacuum the database to reclaim space and optimize performance.

        Returns:
            True if successful

        Raises:
            DatabaseError: If vacuum fails
        """
        try:
            with self.get_connection() as conn:
                # VACUUM requires autocommit mode
                conn.isolation_level = None
                cursor = conn.cursor()
                cursor.execute("VACUUM")
                logger.info("Database vacuumed successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to vacuum database: {str(e)}", exc_info=True)
            raise DatabaseError(f"Vacuum failed: {str(e)}")

    def close(self):
        """Close database connection (for cleanup if needed)."""
        logger.info("Database operations completed")
