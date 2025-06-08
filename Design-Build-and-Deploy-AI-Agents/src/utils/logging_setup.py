import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pyprojroot import here


def setup_logger(name: str = "chatbot") -> logging.Logger:
    """
    Sets up a logger with separate log files for DEBUG, INFO, and ERROR levels.
    Each log file rotates daily at midnight and retains logs for 7 days.

    Args:
        name (str): Name of the logger. Also used as the base filename for logs.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logs_dir = here("logs")
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    levels = ["DEBUG", "INFO", "ERROR"]
    for level in levels:
        handler = TimedRotatingFileHandler(
            filename=os.path.join(logs_dir, f"{name}_{level.lower()}.log"),
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )
        handler.setLevel(getattr(logging, level))
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
