from typing import Optional
import os
import redis
import time

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create a Redis client for helper functions
_redis_client = None

def get_redis_client():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def acquire_lock(key: str, ttl: int = 300) -> bool:
    """Try to acquire a simple Redis lock; returns True if acquired."""
    r = get_redis_client()
    # SET NX with expiration
    return r.set(name=key, value="1", nx=True, ex=ttl)


def release_lock(key: str) -> None:
    """Release a lock (best-effort)."""
    r = get_redis_client()
    try:
        r.delete(key)
    except Exception:
        pass


def with_lock(key: str, ttl: int = 300):
    """Context manager for acquiring and releasing lock.

    Usage:
        with with_lock(f"task-lock:{task_id}", ttl=60):
            # do work
    """
    class _LockCtx:
        def __enter__(self_):
            start = time.time()
            while True:
                if acquire_lock(key, ttl):
                    return True
                if time.time() - start > 5:
                    raise TimeoutError("Failed to acquire lock")
                time.sleep(0.05)
        def __exit__(self_, exc_type, exc, tb):
            release_lock(key)
            return False
    return _LockCtx()
