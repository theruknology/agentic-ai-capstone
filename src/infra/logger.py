import logging
import sys
import time
from functools import wraps

# Create a custom logger
logger = logging.getLogger("J*bLess")
logger.setLevel(logging.INFO)

# Console Handler (Standard Output)
c_handler = logging.StreamHandler(sys.stdout)
c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

# File Handler (Persistent Log)
f_handler = logging.FileHandler("system.log")
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)

def log_latency(func):
    """Decorator to log execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"⏱️ {func.__name__} finished in {end - start:.2f}s")
        return result
    return wrapper