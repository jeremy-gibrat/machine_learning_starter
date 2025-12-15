"""
Fonctions utilitaires
"""

import logging
from pathlib import Path
from config.config import LOG_FILE, LOG_LEVEL


def setup_logger(name='ml_project'):
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
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def safe_divide(numerator, denominator, default=None):
    """
    Division sécurisée avec gestion des erreurs
    
    Args:
        numerator: Numérateur
        denominator: Dénominateur
        default: Valeur par défaut si erreur
        
    Returns:
        float or default: Résultat de la division
    """
    if numerator is None or denominator is None or denominator == 0:
        return default
    try:
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default
