import logging
from typing import List


def to_int(lst: List[str]) -> List[int]:
    """Convert list of strings to ints"""
    try:
        return [int(s) for s in lst]
    except ValueError:
        raise ValueError("All dimensions must be integers!")


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    Args:
        log_path {str} -- where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, "w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)
