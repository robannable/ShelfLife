"""
Data models for ShelfLife application.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

@dataclass
class BookMetadata:
    """Enhanced metadata for a book."""
    synopsis: Optional[str] = None
    genre: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    historical_context: Optional[str] = None
    time_period: Optional[str] = None
    reading_level: Optional[str] = None
    related_works: List[Dict[str, str]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    cover_url: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    year: Optional[int] = None
    publisher: Optional[str] = None
    description: Optional[str] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None and v != [] and v != {}
        }

    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BookMetadata':
        """Create BookMetadata from dictionary."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'BookMetadata':
        """Create BookMetadata from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Book:
    """Represents a book in the collection."""
    title: str
    author: str
    id: Optional[int] = None
    year: Optional[int] = None
    isbn: Optional[str] = None
    publisher: Optional[str] = None
    condition: Optional[str] = "Good"
    cover_image: Optional[bytes] = None
    metadata: Optional[BookMetadata] = None
    personal_notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate book data after initialization."""
        if not self.title or not self.title.strip():
            raise ValueError("Title is required")
        if not self.author or not self.author.strip():
            raise ValueError("Author is required")
        if self.year and (self.year < 0 or self.year > datetime.now().year):
            raise ValueError(f"Invalid year: {self.year}")
        if self.condition and self.condition not in [
            "New", "Like New", "Very Good", "Good", "Fair", "Poor"
        ]:
            raise ValueError(f"Invalid condition: {self.condition}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert book to dictionary for database storage."""
        return {
            'id': self.id,
            'title': self.title,
            'author': self.author,
            'year': self.year,
            'isbn': self.isbn,
            'publisher': self.publisher,
            'condition': self.condition,
            'cover_image': self.cover_image,
            'metadata': self.metadata.to_json() if self.metadata else None,
            'personal_notes': self.personal_notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class LibraryStats:
    """Statistics about the library collection."""
    total_books: int = 0
    unique_authors: int = 0
    avg_pub_year: Optional[int] = None
    common_time_period: Optional[str] = None
    time_period_coverage: float = 0.0
    fiction_ratio: float = 0.0
    nonfiction_ratio: float = 0.0


@dataclass
class ThemeAnalysis:
    """Theme analysis results."""
    uber_themes: List[Dict[str, Any]] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    generated_at: Optional[str] = None


@dataclass
class ExecutiveSummary:
    """Executive summary of the library."""
    summary: str
    patterns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_updated: Optional[str] = None
