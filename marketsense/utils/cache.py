"""
Caching utilities for MarketSense AI.
"""

import os
import json
import time
import hashlib
from functools import wraps

def get_cache_path(cache_key, cache_dir="./data/cache", suffix=".json"):
    """Get path for a cache file with proper sanitization."""
    # Create a safe filename from the cache key
    safe_key = hashlib.md5(cache_key.encode()).hexdigest()
    return os.path.join(cache_dir, f"{safe_key}{suffix}")

def is_cache_valid(cache_path, expiry_days=7):
    """Check if a cache file exists and is still valid."""
    if not os.path.exists(cache_path):
        return False
    
    # Check if cache is still valid
    file_age = time.time() - os.path.getmtime(cache_path)
    return file_age < expiry_days * 86400  # Convert days to seconds

def save_to_cache(data, cache_path):
    """Save data to a cache file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    with open(cache_path, 'w') as f:
        json.dump(data, f)

def load_from_cache(cache_path):
    """Load data from a cache file."""
    with open(cache_path, 'r') as f:
        return json.load(f)

def cached(key_fn=None, expiry_days=7):
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
