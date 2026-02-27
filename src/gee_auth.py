import time
from typing import TypeVar, Callable, Any

import ee

T = TypeVar('T')

def with_retry(max_retries: int = 5, base_delay: float = 2.0, max_delay: float = 60.0):
    """
    Decorator that applies exponential backoff to functions making GEE API calls.
    Useful for handling rate limits and transient network errors (ee.EEException).
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"  [FAIL] Max retries reached for {func.__name__}: {e}")
                        raise
                    delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                    print(f"  [RETRY] GEE error in {func.__name__}: {e}. Retrying in {delay}s... ({retries}/{max_retries})")
                    time.sleep(delay)
        return wrapper
    return decorator


def authenticate_and_initialize() -> None:
    """
    Authenticate and initialize Google Earth Engine.
    
    On first run, this will open a browser for authentication.
    Subsequent runs will use cached credentials.
    """
    try:
        ee.Initialize()
        print("[OK] GEE already authenticated")
    except Exception:
        print("Authenticating GEE...")
        ee.Authenticate()
        ee.Initialize()
        print("[OK] GEE authenticated successfully")


def check_gee_ready() -> bool:
    """Check if GEE is ready to use."""
    try:
        ee.Initialize()
        # Test with a simple operation
        _ = ee.Number(1).getInfo()
        return True
    except Exception as e:
        print(f"[FAIL] GEE not ready: {e}")
        return False
