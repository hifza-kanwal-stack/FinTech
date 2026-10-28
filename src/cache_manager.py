"""
LangChain Caching Configuration (InMemoryCache + SQLiteCache).
"""
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

_MEMORY_CACHE_INSTANCE = None
_SQLITE_CACHE_INSTANCE = None


def configure_cache(cache_type: str, sqlite_path: str = ".langchain_cache.db") -> str:
    """
    Register a global LangChain cache based on user selection.
    Options: 'InMemoryCache', 'SQLiteCache', or 'Disabled'.
    """
    global _MEMORY_CACHE_INSTANCE, _SQLITE_CACHE_INSTANCE

    if cache_type == "InMemoryCache":
        if _MEMORY_CACHE_INSTANCE is None:
            _MEMORY_CACHE_INSTANCE = InMemoryCache()
        set_llm_cache(_MEMORY_CACHE_INSTANCE)
        return "InMemoryCache active (stored in RAM; fastest; resets on restart)."

    if cache_type == "SQLiteCache":
        if _SQLITE_CACHE_INSTANCE is None:
            _SQLITE_CACHE_INSTANCE = SQLiteCache(database_path=sqlite_path)
        set_llm_cache(_SQLITE_CACHE_INSTANCE)
        return f"SQLiteCache active (persisted on disk at '{sqlite_path}')."

    set_llm_cache(None)
    return "LLM Caching is currently disabled."


def clear_active_cache() -> None:
    """Reset in-memory cache instance."""
    global _MEMORY_CACHE_INSTANCE
    _MEMORY_CACHE_INSTANCE = InMemoryCache()
    set_llm_cache(_MEMORY_CACHE_INSTANCE)
