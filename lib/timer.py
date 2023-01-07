"""Timer decorator for timing functions."""
import time


def timer(func):
    """Decorator for timing functions."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        return result
    return wrapper
