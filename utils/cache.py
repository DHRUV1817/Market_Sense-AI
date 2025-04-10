# Caching utilities
"""
Caching utilities for MarketSense AI.
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import wraps
import hashlib

from marketsense import config

def get_cache_path(cache_key: str, suffix: str = ".json") -> Path:
    """Get path for a cache file with proper sanitization."""
    # Create a safe filename from the cache key
    safe_key = hashlib.md5(cache_key.encode()).hexdigest()
    return Path(config.CACHE_DIR) / f"{safe_key}{suffix}"

def is_cache_valid(cache_path: Path, expiry_days: int = config.CACHE_EXPIRY_DAYS) -> bool:
    """Check if a cache file exists and is still valid."""
    if not cache_path.exists():
        return False
    
    # Check if cache is still valid
    file_age = time.time() - cache_path.stat().st_mtime
    return file_age < expiry_days * 86400  # Convert days to seconds

def save_to_cache(data: Any, cache_path: Path) -> None:
    """Save data to a cache file."""
    with open(cache_path, 'w') as f:
        json.dump(data, f)

def load_from_cache(cache_path: Path) -> Any:
    """Load data from a cache file."""
    with open(cache_path, 'r') as f:
        return json.load(f)

def cached(key_fn: Callable = None, expiry_days: int = config.CACHE_EXPIRY_DAYS):
    """
    Decorator to cache function results.
    
    Args:
        key_fn: Function to generate cache key from args and kwargs
        expiry_days: Cache expiry in days
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default cache key based on function name and args
                params = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}_{hash(params)}"
            
            cache_path = get_cache_path(cache_key)
            
            # Return cached result if valid
            if is_cache_valid(cache_path, expiry_days):
                return load_from_cache(cache_path)
            
            # Calculate result and cache it
            result = func(*args, **kwargs)
            save_to_cache(result, cache_path)
            return result
        
        return wrapper
    
    return decorator