import logging
import sys
import os

def get_logger(name: str = "credit_risk", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger instance with standard formatting.
    Avoids duplicate handlers when called multiple times.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Optional: file handler if logs directory exists or can be created
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            logs_dir = os.path.join(base_dir, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(logs_dir, 'credit_risk.log'), encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass  # Fall back to console only if file access is restricted

    return logger
