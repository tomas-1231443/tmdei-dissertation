import logging
import sys
from src import config

def get_logger(name: str) -> logging.Logger:

    """
    Creates and configures a logger with console output.
    :param name: The name of the logger.
    :return: Configured logger instance.
    """

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    level = logging.DEBUG if config.VERBOSE else logging.INFO
    logger.setLevel(level)

    return logger
