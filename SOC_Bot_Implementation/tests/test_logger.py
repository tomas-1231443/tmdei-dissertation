import logging
from src.logger import get_logger
from src import config

def test_get_logger_default():
    # Save the current VERBOSE setting to restore it later
    current_verbose = config.VERBOSE

    # Test logger when VERBOSE is True
    config.VERBOSE = True
    logger_debug = get_logger("test_logger_debug")
    assert isinstance(logger_debug, logging.Logger)
    assert logger_debug.handlers, "Logger should have at least one handler"
    assert logger_debug.level == logging.DEBUG, f"Expected DEBUG level when VERBOSE is True, got {logger_debug.level}"

    # Test logger when VERBOSE is False
    config.VERBOSE = False
    logger_info = get_logger("test_logger_info")
    assert isinstance(logger_info, logging.Logger)
    assert logger_info.handlers, "Logger should have at least one handler"
    assert logger_info.level == logging.INFO, f"Expected INFO level when VERBOSE is False, got {logger_info.level}"

    # Restore the original VERBOSE setting
    config.VERBOSE = current_verbose
