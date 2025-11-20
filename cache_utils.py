"""
Caching utilities for API responses and LLM results.
Implements both in-memory and persistent caching with TTL support.
"""
import json
import hashlib
import time
import sqlite3
from pathlib import Path
from typing import Optional, Any, Callable, Dict, Tuple
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Cache Stats: {self.hits} hits, {self.misses} misses, "
            f"{self.hit_rate:.1f}% hit rate, {self.size} entries"
        )


class InMemoryCache:
    """
    Thread-safe in-memory cache with TTL support.
    Uses LRU eviction when max_size is reached.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self._access_order: Dict[str, float] = {}  # key -> last_access_time
        self.stats = CacheStats()
        logger.info(f"Initialized in-memory cache: max_size={max_size}, ttl={default_ttl}s")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self.stats.misses += 1
            return None

        value, expiry_time = self._cache[key]

        # Check if expired
        if time.time() > expiry_time:
            logger.debug(f"Cache key expired: {key[:50]}")
            self._evict(key)
            self.stats.misses += 1
            return None

        # Update access order for LRU
        self._access_order[key] = time.time()
        self.stats.hits += 1
        logger.debug(f"Cache hit: {key[:50]}")
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Evict oldest entries if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        ttl = ttl or self.default_ttl
        expiry_time = time.time() + ttl

        self._cache[key] = (value, expiry_time)
        self._access_order[key] = time.time()
        self.stats.size = len(self._cache)
        logger.debug(f"Cache set: {key[:50]} (ttl={ttl}s)")

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key

        Returns:
            True if key was found and removed
        """
        if key in self._cache:
            self._evict(key)
            logger.info(f"Cache invalidated: {key[:50]}")
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        self.stats = CacheStats()
        logger.info(f"Cache cleared: {count} entries removed")

    def _evict(self, key: str) -> None:
        """Evict a single cache entry."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            del self._access_order[key]
        self.stats.size = len(self._cache)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return

        # Find LRU key
        lru_key = min(self._access_order, key=self._access_order.get)
        logger.debug(f"Evicting LRU entry: {lru_key[:50]}")
        self._evict(lru_key)

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, (_, expiry) in self._cache.items()
            if now > expiry
        ]

        for key in expired_keys:
            self._evict(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)


class PersistentCache:
    """
    SQLite-based persistent cache with TTL support.
    Survives application restarts.
    """

    def __init__(self, db_path: str = "data/cache.db", default_ttl: int = 86400):
        """
        Initialize persistent cache.

        Args:
            db_path: Path to SQLite database
            default_ttl: Default time-to-live in seconds (default: 24 hours)
        """
        self.db_path = db_path
        self.default_ttl = default_ttl
        self.stats = CacheStats()

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()
        logger.info(f"Initialized persistent cache: {db_path}, ttl={default_ttl}s")

    def _init_db(self) -> None:
        """Initialize cache database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)")
            conn.commit()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

            if not row:
                self.stats.misses += 1
                return None

            value_json, expires_at = row

            # Check if expired
            if time.time() > expires_at:
                logger.debug(f"Persistent cache key expired: {key[:50]}")
                self.invalidate(key)
                self.stats.misses += 1
                return None

            self.stats.hits += 1
            logger.debug(f"Persistent cache hit: {key[:50]}")

            try:
                return json.loads(value_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to deserialize cached value: {e}")
                self.invalidate(key)
                self.stats.misses += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (uses default if None)
        """
        ttl = ttl or self.default_ttl
        now = time.time()
        expires_at = now + ttl

        try:
            value_json = json.dumps(value)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize value for caching: {e}")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, value_json, now, expires_at)
            )
            conn.commit()

        logger.debug(f"Persistent cache set: {key[:50]} (ttl={ttl}s)")
        self._update_stats()

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key

        Returns:
            True if key was found and removed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Persistent cache invalidated: {key[:50]}")
                self._update_stats()
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]

            conn.execute("DELETE FROM cache")
            conn.commit()

        logger.info(f"Persistent cache cleared: {count} entries removed")
        self._update_stats()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
            conn.commit()
            count = cursor.rowcount

        if count > 0:
            logger.info(f"Cleaned up {count} expired persistent cache entries")
            self._update_stats()

        return count

    def _update_stats(self) -> None:
        """Update cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            self.stats.size = cursor.fetchone()[0]


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from function arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        SHA-256 hash of arguments
    """
    # Create a stable string representation of arguments
    key_parts = []

    for arg in args:
        if isinstance(arg, (str, int, float, bool, type(None))):
            key_parts.append(str(arg))
        else:
            # For complex objects, use their string representation
            key_parts.append(repr(arg))

    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool, type(None))):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(f"{k}={repr(v)}")

    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()


def cached(
    cache: Optional[InMemoryCache] = None,
    ttl: Optional[int] = None,
    key_prefix: str = ""
):
    """
    Decorator for caching function results.

    Args:
        cache: Cache instance (creates default if None)
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys

    Example:
        @cached(ttl=3600, key_prefix="api")
        def fetch_data(url):
            return requests.get(url).json()
    """
    # Create default cache if none provided
    if cache is None:
        cache = InMemoryCache()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{generate_cache_key(*args, **kwargs)}"

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = func(*args, **kwargs)

            # Only cache non-None results
            if result is not None:
                cache.set(cache_key, result, ttl=ttl)

            return result

        # Add cache control methods to wrapper
        wrapper.cache = cache
        wrapper.invalidate_cache = lambda *args, **kwargs: cache.invalidate(
            f"{key_prefix}:{func.__name__}:{generate_cache_key(*args, **kwargs)}"
        )
        wrapper.clear_cache = cache.clear

        return wrapper

    return decorator


# Global cache instances
_memory_cache = InMemoryCache(max_size=1000, default_ttl=3600)  # 1 hour default
_persistent_cache = PersistentCache(default_ttl=86400)  # 24 hours default


def get_memory_cache() -> InMemoryCache:
    """Get global in-memory cache instance."""
    return _memory_cache


def get_persistent_cache() -> PersistentCache:
    """Get global persistent cache instance."""
    return _persistent_cache


def get_cache_stats() -> Dict[str, CacheStats]:
    """
    Get statistics for all cache instances.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "memory": _memory_cache.stats,
        "persistent": _persistent_cache.stats
    }


def cleanup_all_caches() -> Dict[str, int]:
    """
    Clean up expired entries from all caches.

    Returns:
        Dictionary with cleanup counts
    """
    return {
        "memory": _memory_cache.cleanup_expired(),
        "persistent": _persistent_cache.cleanup_expired()
    }


def clear_all_caches() -> None:
    """Clear all cache instances."""
    _memory_cache.clear()
    _persistent_cache.clear()
    logger.info("All caches cleared")
