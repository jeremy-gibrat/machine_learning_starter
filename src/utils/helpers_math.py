"""
Fonctions utilitaires mathématiques
"""

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

def round(valeur, decimales=2):
    """
    Arrondit une valeur à un nombre de décimales donné
    
    Args:
        valeur: Valeur à arrondir
        decimales: Nombre de décimales
    Returns:
        float: Valeur arrondie
    """
    try:
        return round(valeur, decimales)
    except (TypeError, ValueError):
        return valeur

def clamp(valeur, minimum, maximum):
    """
    Contraint une valeur entre un minimum et un maximum
    
    Args:
        valeur: Valeur à contraindre
        minimum: Borne inférieure
        maximum: Borne supérieure
    Returns:
        float: Valeur contrainte
    """
    try:
        return max(minimum, min(valeur, maximum))
    except (TypeError, ValueError):
        return valeur

def slope(x1, y1, x2, y2):
    """
    Calcule la pente entre deux points (x1, y1) et (x2, y2)
    """
    try:
        return (y2 - y1) / (x2 - x1)
    except (ZeroDivisionError, TypeError):
        return None

def numerical_derivative(f, x, h=1e-5):
    """
    Calcule la dérivée numérique d'une fonction f en x (méthode des différences centrées)
    """
    try:
        return (f(x + h) - f(x - h)) / (2 * h)
    except Exception:
        return None