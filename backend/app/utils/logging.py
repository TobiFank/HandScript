# backend/app/utils/logging.py
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from ..config import settings

# Ensure logs directory exists
LOG_DIR = settings.STORAGE_PATH / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Create formatters
verbose_formatter = logging.Formatter(
    '\033[1;36m%(asctime)s\033[0m - \033[1;33m%(name)s\033[0m - \033[1;35m%(levelname)s\033[0m [\033[1;34m%(module)s:%(lineno)d\033[0m] - %(message)s'
)
simple_formatter = logging.Formatter(
    '%(levelname)s [%(module)s] - %(message)s'
)

class HandscriptLogger:
    """Custom logger class that protects reserved LogRecord attributes"""
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.setup_handlers()

        # List of reserved LogRecord attributes that shouldn't be overwritten
        self.reserved_attrs = {
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'message', 'msg', 'name', 'pathname', 'process', 'processName',
            'relativeCreated', 'stack_info', 'thread', 'threadName'
        }

    def setup_handlers(self):
        """Set up file and console handlers"""
        if self.logger.handlers:
            return

        # File handler with rotation
        file_handler = RotatingFileHandler(
            LOG_DIR / f"{self.logger.name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(verbose_formatter)
        self.logger.addHandler(file_handler)

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(verbose_formatter)
        self.logger.addHandler(console_handler)

    def _sanitize_extra(self, extra):
        """Sanitize extra fields to avoid conflicts with reserved attributes"""
        if extra is None:
            return None

        sanitized = {}
        for key, value in extra.items():
            if key in self.reserved_attrs:
                safe_key = f"extra_{key}"
                sanitized[safe_key] = value
            else:
                sanitized[key] = value
        return sanitized

    def debug(self, msg, extra=None, exc_info=None):
        self.logger.debug(msg, extra=self._sanitize_extra(extra), exc_info=exc_info)

    def info(self, msg, extra=None, exc_info=None):
        self.logger.info(msg, extra=self._sanitize_extra(extra), exc_info=exc_info)

    def warning(self, msg, extra=None, exc_info=None):
        self.logger.warning(msg, extra=self._sanitize_extra(extra), exc_info=exc_info)

    def error(self, msg, extra=None, exc_info=None):
        self.logger.error(msg, extra=self._sanitize_extra(extra), exc_info=exc_info)

    def critical(self, msg, extra=None, exc_info=None):
        self.logger.critical(msg, extra=self._sanitize_extra(extra), exc_info=exc_info)

# Create loggers for different components
ml_logger = HandscriptLogger("ml")
api_logger = HandscriptLogger("api")
db_logger = HandscriptLogger("database")
service_logger = HandscriptLogger("service")

# Make loggers available at module level
__all__ = ["ml_logger", "api_logger", "db_logger", "service_logger"]