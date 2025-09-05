"""Step caching functionality for Portia steps."""

from __future__ import annotations

import hashlib
import pickle
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from portia.builder.step_v2 import StepV2
    from portia.run_context import RunContext


# Use lru_cache to manage cached step results
# We use a sentinel value to distinguish cache misses from None results
_CACHE_MISS_SENTINEL = b"__CACHE_MISS__"


@lru_cache(maxsize=1000)
def _cached_step_lookup(cache_key: str, cached_result: bytes = _CACHE_MISS_SENTINEL) -> bytes:  # noqa: ARG001
    """Use lru_cache to store and retrieve cached step results.

    Args:
        cache_key: The cache key to lookup/store
        cached_result: The result to cache (uses sentinel value for lookups)

    Returns:
        The cached result bytes

    """
    return cached_result


def _get_cached_step_result(cache_key: str) -> bytes | None:
    """Get cached step result using lru_cache.

    Args:
        cache_key: The cache key to lookup

    Returns:
        Serialized result bytes if found, None otherwise

    """
    result = _cached_step_lookup(cache_key)
    return None if result == _CACHE_MISS_SENTINEL else result


def _set_cached_step_result(cache_key: str, serialized_result: bytes) -> None:
    """Set cached step result using lru_cache.

    Args:
        cache_key: The cache key
        serialized_result: The serialized result bytes

    """
    # Store the result by calling the cached function with the result
    _cached_step_lookup(cache_key, serialized_result)


class StepCache:
    """Handles caching for step results with support for Redis and in-memory caching."""

    # Cache configuration constants
    _REDIS_CACHE_TTL = 3600  # 1 hour

    @staticmethod
    def generate_cache_key(step: StepV2, run_data: RunContext) -> str:
        """Generate a simple MD5-based cache key from step JSON.

        Args:
            step: The step instance to generate cache key for
            run_data: Run context for additional context (plan ID)

        Returns:
            MD5 hash-based cache key

        """
        # Create cache key from step JSON + plan context
        # This captures all step attributes that affect execution
        key_data = {
            "step_json": step.model_dump_json(sort_keys=True),  # Deterministic JSON
            "plan_id": str(run_data.plan.id),
        }

        # Create deterministic string representation
        key_string = f"plan_id:{key_data['plan_id']}|step:{key_data['step_json']}"

        # Create MD5 hash
        key_hash = hashlib.md5(key_string.encode()).hexdigest()  # noqa: S324 # MD5 is fine for cache keys

        return f"step_cache:{key_hash}"

    @staticmethod
    async def get(step: StepV2, run_data: RunContext) -> Any | None:  # noqa: ANN401
        """Get cached result from Redis or in-memory cache.

        Args:
            step: The step to get cached result for
            run_data: Runtime context containing config

        Returns:
            Cached result if found, None otherwise

        """
        # Generate cache key from step
        cache_key = StepCache.generate_cache_key(step, run_data)

        try:
            if run_data.config.step_redis_cache_url:
                return await StepCache._get_from_redis_cache(
                    cache_key, run_data.config.step_redis_cache_url
                )
            return await StepCache._get_from_memory_cache(cache_key)
        except (ImportError, ConnectionError, OSError, AttributeError):
            # If cache retrieval fails due to Redis/connection issues, continue without cache
            return None

    @staticmethod
    async def set(step: StepV2, result: Any, run_data: RunContext) -> None:  # noqa: ANN401
        """Set result in Redis or in-memory cache.

        Args:
            step: The step to set cached result for
            result: The result to cache
            run_data: Runtime context containing config

        """
        # Generate cache key from step
        cache_key = StepCache.generate_cache_key(step, run_data)

        try:
            if run_data.config.step_redis_cache_url:
                await StepCache._set_in_redis_cache(
                    cache_key, result, run_data.config.step_redis_cache_url
                )
            else:
                await StepCache._set_in_memory_cache(cache_key, result)
        except (ImportError, ConnectionError, OSError, AttributeError):
            # If cache setting fails due to Redis/connection issues, continue without caching
            pass

    @staticmethod
    async def _get_from_redis_cache(cache_key: str, redis_url: str) -> Any | None:  # noqa: ANN401
        """Get result from Redis cache."""
        try:
            import redis.asyncio as redis
        except ImportError:
            # Redis not available, fall back to memory cache
            return await StepCache._get_from_memory_cache(cache_key)

        try:
            async with redis.from_url(redis_url) as redis_client:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    return pickle.loads(cached_data)  # noqa: S301 # Pickle is needed for caching serialized objects
                return None
        except (ConnectionError, OSError, AttributeError):
            # If Redis fails, fall back to memory cache
            return await StepCache._get_from_memory_cache(cache_key)

    @staticmethod
    async def _set_in_redis_cache(cache_key: str, result: Any, redis_url: str) -> None:  # noqa: ANN401
        """Set result in Redis cache."""
        try:
            import redis.asyncio as redis
        except ImportError:
            # Redis not available, fall back to memory cache
            await StepCache._set_in_memory_cache(cache_key, result)
            return

        try:
            async with redis.from_url(redis_url) as redis_client:
                # Cache for specified TTL
                serialized_data = pickle.dumps(result)
                await redis_client.setex(cache_key, StepCache._REDIS_CACHE_TTL, serialized_data)
        except (ConnectionError, OSError, AttributeError):
            # If Redis fails, fall back to memory cache
            await StepCache._set_in_memory_cache(cache_key, result)

    @staticmethod
    async def _get_from_memory_cache(cache_key: str) -> Any | None:  # noqa: ANN401
        """Get result from in-memory cache."""
        cached_data = _get_cached_step_result(cache_key)
        if cached_data:
            return pickle.loads(cached_data)  # noqa: S301 # Pickle is needed for caching serialized objects
        return None

    @staticmethod
    async def _set_in_memory_cache(cache_key: str, result: Any) -> None:  # noqa: ANN401
        """Set result in in-memory cache."""
        try:
            serialized_data = pickle.dumps(result)
            _set_cached_step_result(cache_key, serialized_data)
        except (TypeError, AttributeError):
            # If serialization fails, don't cache
            pass
