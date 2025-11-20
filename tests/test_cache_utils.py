"""
Tests for caching utilities.
"""
import pytest
import time
import json
from pathlib import Path
from cache_utils import (
    InMemoryCache,
    PersistentCache,
    CacheStats,
    generate_cache_key,
    cached,
    get_memory_cache,
    get_persistent_cache,
    cleanup_all_caches,
    clear_all_caches
)


class TestCacheStats:
    """Test cache statistics."""

    def test_initial_stats(self):
        """Test initial statistics values."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.size == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 70.0

    def test_hit_rate_no_requests(self):
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_stats_string(self):
        """Test string representation."""
        stats = CacheStats(hits=10, misses=5, size=20)
        assert "10 hits" in str(stats)
        assert "5 misses" in str(stats)
        assert "66.7%" in str(stats)


class TestInMemoryCache:
    """Test in-memory cache."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = InMemoryCache(max_size=100, default_ttl=60)
        assert cache.max_size == 100
        assert cache.default_ttl == 60
        assert cache.stats.size == 0

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = InMemoryCache()
        cache.set("key1", "value1")

        value = cache.get("key1")
        assert value == "value1"
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0

    def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        cache = InMemoryCache()
        value = cache.get("nonexistent")

        assert value is None
        assert cache.stats.misses == 1

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = InMemoryCache(default_ttl=1)
        cache.set("key1", "value1", ttl=1)

        # Should be available immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("key1") is None
        assert cache.stats.misses == 1

    def test_custom_ttl(self):
        """Test custom TTL for specific entry."""
        cache = InMemoryCache(default_ttl=10)
        cache.set("short_ttl", "value", ttl=1)
        cache.set("long_ttl", "value", ttl=100)

        time.sleep(1.1)

        assert cache.get("short_ttl") is None
        assert cache.get("long_ttl") == "value"

    def test_lru_eviction(self):
        """Test LRU eviction when max size reached."""
        cache = InMemoryCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_update_existing_key(self):
        """Test updating existing key."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key1", "value2")

        assert cache.get("key1") == "value2"
        assert cache.stats.size == 1

    def test_invalidate(self):
        """Test cache invalidation."""
        cache = InMemoryCache()
        cache.set("key1", "value1")

        result = cache.invalidate("key1")
        assert result is True
        assert cache.get("key1") is None

        # Invalidating non-existent key
        result = cache.invalidate("nonexistent")
        assert result is False

    def test_clear(self):
        """Test clearing all cache entries."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.stats.size == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = InMemoryCache()
        cache.set("expired1", "value1", ttl=1)
        cache.set("expired2", "value2", ttl=1)
        cache.set("valid", "value3", ttl=100)

        time.sleep(1.1)

        count = cache.cleanup_expired()

        assert count == 2
        assert cache.get("valid") == "value3"
        assert cache.stats.size == 1

    def test_complex_values(self):
        """Test caching complex data types."""
        cache = InMemoryCache()

        # Dictionary
        cache.set("dict", {"key": "value", "nested": {"foo": "bar"}})
        assert cache.get("dict") == {"key": "value", "nested": {"foo": "bar"}}

        # List
        cache.set("list", [1, 2, 3, {"a": "b"}])
        assert cache.get("list") == [1, 2, 3, {"a": "b"}]

        # Tuple
        cache.set("tuple", (1, 2, 3))
        assert cache.get("tuple") == (1, 2, 3)


class TestPersistentCache:
    """Test persistent cache."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database path."""
        db_path = tmp_path / "test_cache.db"
        yield str(db_path)
        # Cleanup
        if db_path.exists():
            db_path.unlink()

    def test_initialization(self, temp_db):
        """Test cache initialization."""
        cache = PersistentCache(db_path=temp_db, default_ttl=60)
        assert cache.db_path == temp_db
        assert cache.default_ttl == 60
        assert Path(temp_db).exists()

    def test_set_and_get(self, temp_db):
        """Test basic set and get operations."""
        cache = PersistentCache(db_path=temp_db)
        cache.set("key1", {"data": "value1"})

        value = cache.get("key1")
        assert value == {"data": "value1"}
        assert cache.stats.hits == 1

    def test_persistence_across_instances(self, temp_db):
        """Test data persists across cache instances."""
        cache1 = PersistentCache(db_path=temp_db)
        cache1.set("key1", {"data": "value1"})

        # Create new instance
        cache2 = PersistentCache(db_path=temp_db)
        value = cache2.get("key1")

        assert value == {"data": "value1"}

    def test_ttl_expiration(self, temp_db):
        """Test TTL expiration."""
        cache = PersistentCache(db_path=temp_db, default_ttl=1)
        cache.set("key1", "value1", ttl=1)

        # Should be available immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("key1") is None

    def test_json_serialization(self, temp_db):
        """Test JSON serialization of values."""
        cache = PersistentCache(db_path=temp_db)

        # Complex nested structure
        data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "nested": {
                "a": "b",
                "c": [4, 5, 6]
            }
        }

        cache.set("complex", data)
        retrieved = cache.get("complex")

        assert retrieved == data

    def test_invalidate(self, temp_db):
        """Test cache invalidation."""
        cache = PersistentCache(db_path=temp_db)
        cache.set("key1", "value1")

        result = cache.invalidate("key1")
        assert result is True
        assert cache.get("key1") is None

        # Invalidating non-existent key
        result = cache.invalidate("nonexistent")
        assert result is False

    def test_clear(self, temp_db):
        """Test clearing all cache entries."""
        cache = PersistentCache(db_path=temp_db)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cleanup_expired(self, temp_db):
        """Test cleanup of expired entries."""
        cache = PersistentCache(db_path=temp_db)
        cache.set("expired1", "value1", ttl=1)
        cache.set("expired2", "value2", ttl=1)
        cache.set("valid", "value3", ttl=100)

        time.sleep(1.1)

        count = cache.cleanup_expired()

        assert count == 2
        assert cache.get("valid") == "value3"


class TestGenerateCacheKey:
    """Test cache key generation."""

    def test_simple_args(self):
        """Test key generation with simple arguments."""
        key1 = generate_cache_key("arg1", "arg2")
        key2 = generate_cache_key("arg1", "arg2")
        key3 = generate_cache_key("arg1", "different")

        # Same args should produce same key
        assert key1 == key2

        # Different args should produce different key
        assert key1 != key3

    def test_with_kwargs(self):
        """Test key generation with keyword arguments."""
        key1 = generate_cache_key("arg1", foo="bar", baz="qux")
        key2 = generate_cache_key("arg1", foo="bar", baz="qux")
        key3 = generate_cache_key("arg1", foo="different", baz="qux")

        assert key1 == key2
        assert key1 != key3

    def test_order_independence(self):
        """Test that kwargs order doesn't matter."""
        key1 = generate_cache_key(foo="bar", baz="qux")
        key2 = generate_cache_key(baz="qux", foo="bar")

        # Should be the same (kwargs are sorted)
        assert key1 == key2

    def test_different_types(self):
        """Test key generation with different types."""
        key1 = generate_cache_key("string", 42, 3.14, True, None)
        key2 = generate_cache_key("string", 42, 3.14, True, None)

        assert key1 == key2

    def test_deterministic(self):
        """Test that key generation is deterministic."""
        keys = [generate_cache_key("test", "args", num=42) for _ in range(10)]
        assert len(set(keys)) == 1  # All keys should be the same


class TestCachedDecorator:
    """Test cached decorator."""

    def test_basic_caching(self):
        """Test basic function caching."""
        call_count = 0

        @cached(ttl=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call - should execute
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Not called again

        # Third call with different args - should execute
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    def test_cache_invalidation(self):
        """Test cache invalidation via decorator."""
        call_count = 0

        @cached(ttl=60, key_prefix="test")
        def my_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Call function
        my_function(5)
        assert call_count == 1

        # Call again - cached
        my_function(5)
        assert call_count == 1

        # Invalidate cache
        my_function.invalidate_cache(5)

        # Call again - should execute
        my_function(5)
        assert call_count == 2

    def test_none_values_not_cached(self):
        """Test that None values are not cached."""
        call_count = 0

        @cached(ttl=60)
        def returns_none(x):
            nonlocal call_count
            call_count += 1
            return None

        # Call multiple times
        returns_none(1)
        returns_none(1)
        returns_none(1)

        # Should be called every time (None not cached)
        assert call_count == 3

    def test_custom_cache_instance(self):
        """Test using custom cache instance."""
        custom_cache = InMemoryCache(max_size=10, default_ttl=30)

        @cached(cache=custom_cache, ttl=60)
        def my_function(x):
            return x * 2

        result = my_function(5)
        assert result == 10
        assert custom_cache.stats.size > 0


class TestGlobalFunctions:
    """Test global cache management functions."""

    def test_get_memory_cache(self):
        """Test getting global memory cache."""
        cache = get_memory_cache()
        assert isinstance(cache, InMemoryCache)

        # Should return same instance
        cache2 = get_memory_cache()
        assert cache is cache2

    def test_get_persistent_cache(self):
        """Test getting global persistent cache."""
        cache = get_persistent_cache()
        assert isinstance(cache, PersistentCache)

        # Should return same instance
        cache2 = get_persistent_cache()
        assert cache is cache2

    def test_cleanup_all_caches(self):
        """Test cleanup of all caches."""
        mem_cache = get_memory_cache()
        persist_cache = get_persistent_cache()

        # Add some entries with short TTL
        mem_cache.set("expired", "value", ttl=1)
        persist_cache.set("expired", "value", ttl=1)

        time.sleep(1.1)

        counts = cleanup_all_caches()

        assert "memory" in counts
        assert "persistent" in counts

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        mem_cache = get_memory_cache()
        persist_cache = get_persistent_cache()

        # Add some entries
        mem_cache.set("key1", "value1")
        persist_cache.set("key2", "value2")

        clear_all_caches()

        assert mem_cache.get("key1") is None
        assert persist_cache.get("key2") is None
