import logging
import logging.handlers
import sys
from pathlib import Path

class SimpleLogger:
    """Simple logger with console and file output (UTF-8 safe)."""
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        console_level: str = "DEBUG",
        file_level: str = "DEBUG"
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        Path(log_dir).mkdir(exist_ok=True)
        
        # Console Handler (force UTF-8)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level))
        console_format = logging.Formatter(
            '\033[36m%(asctime)s\033[0m | \033[32m%(levelname)-8s\033[0m | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File Handler (with UTF-8)
        file_handler = logging.handlers.RotatingFileHandler(
            f"{log_dir}/{name}.log",
            maxBytes=10*1024*1024,  
            backupCount=3,
            encoding="utf-8"  # 👈 quan trọng
        )
        file_handler.setLevel(getattr(logging, file_level))
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def debug(self, msg: str): self.logger.debug(msg)
    def info(self, msg: str): self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str): self.logger.error(msg)
    def critical(self, msg: str): self.logger.critical(msg)
    def exception(self, msg: str): self.logger.exception(msg)
