import logging
from logging.handlers import RotatingFileHandler
import sys
from src import config
from functools import wraps
import os
import time

def get_logger(name: str) -> logging.Logger:

    """
    Creates and configures a logger with console output.    
    :param name: The name of the logger.
    :return: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)

        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        formatter.converter = time.gmtime
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = RotatingFileHandler(f"logs/{name}.log", maxBytes=10**6, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    level = logging.DEBUG if config.VERBOSE else logging.INFO
    logger.setLevel(level)

    return logger

def with_logger(func):
    """
    Decorator that injects a logger into the decorated function.
    The logger will have a fully qualified name (module + function name).
    The decorated function must accept a keyword argument 'logger'.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a logger with the fully qualified name
        logger_name = f"{func.__module__}.{func.__name__}"
        kwargs.setdefault("logger", get_logger(logger_name))
        return func(*args, **kwargs)
    return wrapper
