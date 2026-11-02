import logging
import sys
from pathlib import Path

def setup_logger(log_file: str = "conversion.log") -> logging.Logger:
    """
    Sets up a thread-safe logger that outputs to both a file and the console.
    """
    logger = logging.getLogger("ABIDE_Converter")
    logger.setLevel(logging.INFO)

    # To avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()
