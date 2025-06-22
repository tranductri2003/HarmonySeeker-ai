import logging
import os
import json
import traceback
import sys
from logging.handlers import RotatingFileHandler


class SingleLineExceptionFormatter(logging.Formatter):
    """
    Custom formatter to convert multi-line traceback to single line
    """

    def formatException(self, exc_info):
        """
        Format an exception so that it's traceback is on a single line
        """
        result = super().formatException(exc_info)
        return repr(result)  # Convert to a single line

    def format(self, record):
        """
        Format the record with single line exception
        """
        s = super().format(record)
        if record.exc_text:
            # Convert multi-line traceback to single line
            s = s.replace("\n", " | ")
        return s


def setup_logger(log_file="app.log", log_level=logging.INFO):
    """
    Set up a logger that writes to both console and file

    Args:
        log_file (str): Path to the log file
        log_level: Logging level (default: INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("harmony_seeker")
    logger.setLevel(log_level)

    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    file_formatter = SingleLineExceptionFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    console_formatter = SingleLineExceptionFormatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create file handler for logging to file (with rotation)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,  # Keep 5 backup copies
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)

    # Create console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_exception(logger, message, exc_info=True, **extra):
    """
    Log an exception with traceback on a single line

    Args:
        logger: Logger instance
        message: Message to log
        exc_info: Whether to include exception info
        **extra: Extra fields to include in the log
    """
    if exc_info:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_type is not None:
            tb_str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
            # Replace newlines with pipe symbol
            tb_str = tb_str.replace("\n", " | ")
            message = f"{message} | {tb_str}"

    # Add extra fields
    if extra:
        extra_str = json.dumps(extra)
        message = f"{message} | {extra_str}"

    logger.error(message)
