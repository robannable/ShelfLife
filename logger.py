"""
Centralized logging configuration for ShelfLife application.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

class ShelfLifeLogger:
    """Centralized logger for the application."""

    _instance = None
    _loggers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ShelfLifeLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging with both file and console handlers."""
        # Create logs directory
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        log_file = log_dir / f"shelflife_{datetime.now().strftime('%Y%m%d')}.log"

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger for a specific module."""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]

    def set_level(self, level: str):
        """Set logging level for all loggers."""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        logging.getLogger().setLevel(level_map.get(level.upper(), logging.INFO))

def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger."""
    return ShelfLifeLogger().get_logger(name)
