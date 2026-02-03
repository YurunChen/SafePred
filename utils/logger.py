"""
Unified logging configuration for SafePred.

Provides a centralized logging setup with clear, concise log messages and color support.
"""

import logging
import sys
import os
import threading
from datetime import datetime
from typing import Optional
from pathlib import Path


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    # Reset
    RESET = '\033[0m'
    
    # Level colors
    INFO = '\033[32m'      # Green
    WARNING = '\033[33m'   # Yellow
    ERROR = '\033[31m'     # Red
    CRITICAL = '\033[35m'  # Magenta
    DEBUG = '\033[36m'     # Cyan


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for console output."""
    
    def __init__(self, use_color: bool = True, *args, **kwargs):
        """
        Initialize colored formatter.
        
        Args:
            use_color: Whether to use colors (default: True)
            *args, **kwargs: Arguments passed to logging.Formatter
        """
        super().__init__(*args, **kwargs)
        self.use_color = use_color and sys.stdout.isatty()  # Only use color if terminal supports it
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.
        
        Args:
            record: Log record to format
        
        Returns:
            Formatted log message with colors (if enabled)
        """
        # Get the original formatted message
        message = super().format(record)
        
        # Add colors based on log level (only for console output)
        if self.use_color:
            if record.levelno == logging.INFO:
                # Replace [INFO] with green [INFO]
                message = message.replace('[INFO]', f'{Colors.INFO}[INFO]{Colors.RESET}')
            elif record.levelno == logging.WARNING:
                # Replace [WARNING] with yellow [WARNING]
                message = message.replace('[WARNING]', f'{Colors.WARNING}[WARNING]{Colors.RESET}')
            elif record.levelno == logging.ERROR:
                # Replace [ERROR] with red [ERROR]
                message = message.replace('[ERROR]', f'{Colors.ERROR}[ERROR]{Colors.RESET}')
            elif record.levelno == logging.CRITICAL:
                # Replace [CRITICAL] with magenta [CRITICAL]
                message = message.replace('[CRITICAL]', f'{Colors.CRITICAL}[CRITICAL]{Colors.RESET}')
            elif record.levelno == logging.DEBUG:
                # Replace [DEBUG] with cyan [DEBUG]
                message = message.replace('[DEBUG]', f'{Colors.DEBUG}[DEBUG]{Colors.RESET}')
        
        return message


class SafePredLogger:
    """Unified logger for SafePred package."""
    
    _logger: Optional[logging.Logger] = None
    _configured: bool = False
    _log_file_path: Optional[str] = None
    _root_configured: bool = False  # Track if root logger is configured
    _lock = threading.Lock()  # Lock for thread-safe initialization
    _web_agent_model_name: Optional[str] = None  # Web agent model name for log filename
    _world_model_name: Optional[str] = None  # World model name for log filename
    
    @classmethod
    def get_logger(cls, name: str = "SafePred") -> logging.Logger:
        """
        Get or create the logger instance.
        
        Args:
            name: Logger name (default: "SafePred")
        
        Returns:
            Configured logger instance
        """
        # Configure root logger only once (thread-safe)
        if not cls._root_configured:
            with cls._lock:
                # Double-check after acquiring lock
                if not cls._root_configured:
                    cls._setup_root_logger()
        
        # Return the specific logger instance
        return logging.getLogger(name)
    
    @classmethod
    def get_log_file_path(cls) -> Optional[str]:
        """
        Get the current log file path.
        
        Returns:
            Path to the log file, or None if not configured
        """
        return cls._log_file_path
    
    @classmethod
    def set_model_names(cls, web_agent_model_name: Optional[str] = None, world_model_name: Optional[str] = None) -> None:
        """
        Set model names for log filename.
        
        If logger is already configured, this will update the log file name.
        
        Args:
            web_agent_model_name: Web agent model name
            world_model_name: World model name
        """
        cls._web_agent_model_name = web_agent_model_name
        cls._world_model_name = world_model_name
        
        # If logger is already configured, update the log file name
        if cls._root_configured:
            cls._update_log_filename()
    
    @staticmethod
    def _sanitize_model_name_for_file(model_name: Optional[str]) -> Optional[str]:
        """
        Sanitize model name for use in log filename.
        
        Args:
            model_name: Original model name (can be None)
            
        Returns:
            Sanitized model name safe for filename use, or None
        """
        if model_name is None:
            return None
        import re
        # Remove or replace invalid filename characters
        # Replace slashes, colons, and other special chars with underscores
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', model_name)
        # Replace multiple underscores with single underscore
        safe_name = re.sub(r'_+', '_', safe_name)
        # Remove leading/trailing underscores and dots
        safe_name = safe_name.strip('_.')
        # Limit length to avoid filename issues
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        return safe_name if safe_name else None
    
    @classmethod
    def _update_log_filename(cls) -> None:
        """
        Update log filename to include model names.
        This creates a new log file with the updated name.
        """
        root_logger = logging.getLogger("SafePred")
        
        # Remove existing file handler
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        for handler in file_handlers:
            handler.close()
            root_logger.removeHandler(handler)
        
        # Create new log file with model names
        try:
            # Get SafePred root (logger.py is in SafePred/utils/)
            safepred_root = Path(__file__).parent.parent
            logs_base_dir = safepred_root / "logs"
            
            # Create date-based subdirectory (YYYY-MM-DD format)
            date_str = datetime.now().strftime("%Y-%m-%d")
            logs_dir = logs_base_dir / date_str
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp filename (YYYYMMDD_HHMMSS format)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Include model names in filename
            model_suffix = ""
            if cls._web_agent_model_name or cls._world_model_name:
                # Sanitize model names for filename
                web_agent = cls._sanitize_model_name_for_file(cls._web_agent_model_name) or "unknown_web_agent"
                world_model = cls._sanitize_model_name_for_file(cls._world_model_name) or "unknown_world_model"
                model_suffix = f"_{web_agent}_{world_model}"
            
            log_filename = f"safepred_{timestamp}{model_suffix}.log"
            cls._log_file_path = str(logs_dir / log_filename)
            
            # Get log level from config, environment variable, or default to INFO
            # Priority: config > environment variable > default
            log_level_str = None
            try:
                # Try to get from config if available
                from ..config.config import SafetyConfig
                try:
                    config = SafetyConfig.from_yaml()
                    if hasattr(config, 'log_level') and config.log_level:
                        log_level_str = config.log_level.upper()
                except Exception:
                    pass  # Config not available, fall back to environment variable
            except ImportError:
                pass  # Config module not available, fall back to environment variable
            
            # Fall back to environment variable if config not available
            if log_level_str is None:
                log_level_str = os.getenv('SAFEPRED_LOG_LEVEL', 'INFO').upper()
            
            log_level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL,
            }
            log_level = log_level_map.get(log_level_str, logging.INFO)
            
            file_handler = logging.FileHandler(cls._log_file_path, encoding='utf-8')
            file_handler.setLevel(log_level)
            # File handler uses plain formatter (no colors)
            base_format = '[%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
            file_formatter = logging.Formatter(
                fmt=base_format,
                datefmt='%H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            root_logger.info(f"Updated log file to: {cls._log_file_path} (log level: {log_level_str})")
        except Exception as e:
            root_logger.warning(f"Failed to update log file: {e}")
    
    @classmethod
    def _setup_root_logger(cls) -> None:
        """
        Setup root logger configuration (only called once).
        This method should only be called while holding cls._lock.
        """
        # Get root logger "SafePred" to configure all child loggers
        root_logger = logging.getLogger("SafePred")
        
        # Check if already configured (by another thread/import)
        if root_logger.handlers:
            cls._root_configured = True
            return
        
        # Get log level from config, environment variable, or default to INFO
        # Priority: config > environment variable > default
        log_level_str = None
        try:
            # Try to get from config if available
            from ..config.config import SafetyConfig
            # Try to load config (may fail if config not initialized yet)
            try:
                config = SafetyConfig.from_yaml()
                if hasattr(config, 'log_level') and config.log_level:
                    log_level_str = config.log_level.upper()
            except Exception:
                pass  # Config not available, fall back to environment variable
        except ImportError:
            pass  # Config module not available, fall back to environment variable
        
        # Fall back to environment variable if config not available
        if log_level_str is None:
            log_level_str = os.getenv('SAFEPRED_LOG_LEVEL', 'INFO').upper()
        
        log_level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        log_level = log_level_map.get(log_level_str, logging.INFO)
        
        root_logger.setLevel(log_level)
        
        # Format with file location: [LEVEL] filename:line - message
        base_format = '[%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
        
        # Create console handler with colored formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = ColoredFormatter(
            use_color=True,
            fmt=base_format,
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Create file handler with plain formatter (no colors in file)
        try:
            # Get SafePred root (logger.py is in SafePred/utils/)
            safepred_root = Path(__file__).parent.parent
            logs_base_dir = safepred_root / "logs"
            
            # Create date-based subdirectory (YYYY-MM-DD format)
            date_str = datetime.now().strftime("%Y-%m-%d")
            logs_dir = logs_base_dir / date_str
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp filename (YYYYMMDD_HHMMSS format)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Include model names in filename if available
            model_suffix = ""
            if cls._web_agent_model_name or cls._world_model_name:
                # Sanitize model names for filename
                web_agent = cls._sanitize_model_name_for_file(cls._web_agent_model_name) or "unknown_web_agent"
                world_model = cls._sanitize_model_name_for_file(cls._world_model_name) or "unknown_world_model"
                model_suffix = f"_{web_agent}_{world_model}"
            
            log_filename = f"safepred_{timestamp}{model_suffix}.log"
            cls._log_file_path = str(logs_dir / log_filename)
            
            # Use the same log level as console handler
            file_handler = logging.FileHandler(cls._log_file_path, encoding='utf-8')
            file_handler.setLevel(log_level)
            # File handler uses plain formatter (no colors)
            file_formatter = logging.Formatter(
                fmt=base_format,
                datefmt='%H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            # Mark as configured BEFORE logging to avoid recursive calls
            cls._root_configured = True
            
            # Now safe to log
            root_logger.info(f"Logging to file: {cls._log_file_path} (log level: {log_level_str})")
        except Exception as e:
            # If file logging fails, continue with console only
            root_logger.warning(f"Failed to setup file logging: {e}")
        
        # Disable propagation to avoid duplicate logs
        root_logger.propagate = False
        
        # Ensure all existing child loggers also have propagate=False
        # This prevents any loggers created before root configuration from propagating
        for logger_name in logging.Logger.manager.loggerDict:
            if isinstance(logging.Logger.manager.loggerDict[logger_name], logging.Logger):
                child_logger = logging.Logger.manager.loggerDict[logger_name]
                if child_logger.name.startswith("SafePred") and child_logger != root_logger:
                    child_logger.propagate = False
        
        # Mark as configured (already set before logging, but ensure it's set)
        cls._root_configured = True
    
    @classmethod
    def set_level(cls, level: str) -> None:
        """
        Set logging level.
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        # Ensure root logger is configured (thread-safe)
        if not cls._root_configured:
            with cls._lock:
                if not cls._root_configured:
                    cls._setup_root_logger()
        
        root_logger = logging.getLogger("SafePred")
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        
        root_logger.setLevel(level_map.get(level.upper(), logging.INFO))
        for handler in root_logger.handlers:
            handler.setLevel(level_map.get(level.upper(), logging.INFO))


def get_logger(name: str = "SafePred") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name (default: "SafePred")
    
    Returns:
        Logger instance
    
    Example:
        logger = get_logger("SafePred.WorldModel")
        logger.error("Failed to simulate state")
        logger.info("State simulation completed")
    """
    return SafePredLogger.get_logger(name)

