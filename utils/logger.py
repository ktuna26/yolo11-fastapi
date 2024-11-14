""" utils/loger.py
Logger, it is used to log 
the process about the inference pipline

Copyright 2024 ktun@

CREATED:  2024-11-12 23:12:13
MODIFIED: 2024-11-1316:30:45
"""
# -*- coding:utf-8 -*-
# Import the necessary libraries
import os
import logging
#
from enum import Enum
from rich.logging import RichHandler


class LoggerLevels(Enum):
    """Define standard logging levels with an enum for clear references."""
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING


def console_handler():
    """Create a console handler with RichHandler for colorful and structured output."""
    msg = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
    console_handler = RichHandler(markup=True)
    console_handler.setFormatter(logging.Formatter(msg, datefmt="%Y-%m-%d %H:%M:%S"))
    return console_handler

def file_handler(log_file='app.log', max_bytes=5 * 1024 * 1024, backup_count=5):
    """Create a rotating file handler for persistent logging to file.
    
    Parameters:
    - log_file : str : Path to the log file.
    - max_bytes : int : Maximum size of a log file before rotating.
    - backup_count : int : Number of backup files to keep.
    """
    msg = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(msg, datefmt="%Y-%m-%d %H:%M:%S"))
    return file_handler

def get_logger(name: str, verbosity: str = "INFO", log_file: str = 'app.log') -> logging.Logger:
    """
    Create or retrieve a logger with console and file handlers.

    Parameters:
    - name : str : Name of the logger, typically the module's __name__.
    - verbosity : str : Logging level, default is 'INFO'.
    - log_file : str : File path for the rotating file handler.

    Returns:
    - logging.Logger : Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Convert verbosity level to uppercase for consistency
    verbosity = verbosity.upper()

    # Only set up handlers if they haven't been added yet
    if not logger.hasHandlers():
        logger.setLevel(LoggerLevels[verbosity].value)
        
        # Add console handler
        logger.addHandler(console_handler())
        
        # Add file handler
        logger.addHandler(file_handler(log_file=log_file))

        # Ensure logs do not propagate to root logger
        logger.propagate = False

    return logger