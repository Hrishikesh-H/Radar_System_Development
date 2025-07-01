# system_logger.py
import logging
import os
from logging.handlers import RotatingFileHandler

class SystemLogger:
    def __init__(self, log_dir="logs", 
                 console_level=logging.WARNING,
                 file_level=logging.DEBUG,
                 max_bytes=10*1024*1024, 
                 backup_count=5):
        self.log_dir = log_dir
        self.console_level = console_level
        self.file_level = file_level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure root logger
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)  # Capture all messages
        
        # Console handler with configurable level
        self.console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.console_handler.setFormatter(console_format)
        self.console_handler.setLevel(console_level)
        self.root_logger.addHandler(self.console_handler)
        
        # File handler with rotation and full detail
        log_file = os.path.join(self.log_dir, "system.log")
        self.file_handler = RotatingFileHandler(
            log_file, maxBytes=self.max_bytes, backupCount=self.backup_count
        )
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.file_handler.setFormatter(file_format)
        self.file_handler.setLevel(file_level)
        self.root_logger.addHandler(self.file_handler)
    
    def get_logger(self, name):
        """Get a named logger for a subsystem"""
        logger = logging.getLogger(name)
        # Don't propagate to root to avoid duplicate logs
        logger.propagate = False
        
        # Add our custom handlers directly to this logger
        logger.addHandler(self.console_handler)
        logger.addHandler(self.file_handler)
        
        # Set logger to capture all levels, let handlers filter
        logger.setLevel(logging.DEBUG)
        
        return logger
    
    def set_console_level(self, level):
        """Dynamically change console log level"""
        self.console_handler.setLevel(level)
        logging.info(f"Console log level changed to {logging.getLevelName(level)}")

# Global logger instance (singleton)
system_logger = None

def init_logger(log_dir="logs", console_level=logging.WARNING, file_level=logging.DEBUG):
    """Initialize the global logger"""
    global system_logger
    if system_logger is None:
        system_logger = SystemLogger(
            log_dir=log_dir,
            console_level=console_level,
            file_level=file_level
        )
    return system_logger

def get_logger(name):
    """Get a named logger for a subsystem"""
    if system_logger is None:
        init_logger()
    return system_logger.get_logger(name)

def set_console_log_level(level):
    """Change console log level globally"""
    if system_logger:
        system_logger.set_console_level(level)
    else:
        logging.warning("Logger not initialized, call init_logger() first")