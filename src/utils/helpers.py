"""
Fonctions utilitaires
"""

import logging
from pathlib import Path
from config.config import LOG_FILE, LOG_LEVEL

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Vert
        'WARNING': '\033[33m',  # Jaune
        'ERROR': '\033[31m',    # Rouge
        'CRITICAL': '\033[41m', # Fond rouge
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        message = super().format(record)
        return f"{color}{message}{self.RESET}"
    

def setup_logger(name='ml_project', colored=True):
    """
    Configure le logger pour le projet
    
    Args:
        name (str): Nom du logger
        
    Returns:
        logger: Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Handler pour fichier
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(LOG_LEVEL)
    
    # Handler pour console
    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    if colored:
        color_formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(color_formatter)
    else:
        ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

