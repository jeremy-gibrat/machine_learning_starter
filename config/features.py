"""
Définition des features pour le modèle ML
"""

# ============================================================================
# FEATURE NAMES
# ============================================================================

# Features numériques de base
NUMERIC_FEATURES = [
    'feature_1',
    'feature_2',
    'feature_3',
]

# Features catégorielles
CATEGORICAL_FEATURES = [
    'category_a',
    'category_b',
]

# Features calculées / dérivées
DERIVED_FEATURES = [
    'ratio_feature_1_2',
    'mean_feature_3',
]

# Toutes les features pour le modèle
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + DERIVED_FEATURES

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def compute_derived_features(df):
    """
    Calcule les features dérivées à partir des features de base
    
    Args:
        df: DataFrame avec les features de base
        
    Returns:
        DataFrame avec les features dérivées ajoutées
    """
    # Exemple: ratio entre deux features
    if 'feature_1' in df.columns and 'feature_2' in df.columns:
        df['ratio_feature_1_2'] = df['feature_1'] / df['feature_2'].replace(0, 1)
    
    # Exemple: moyenne mobile
    if 'feature_3' in df.columns:
        df['mean_feature_3'] = df['feature_3'].rolling(window=10).mean()
    
    return df


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_features(df):
    """
    Valide que toutes les features requises sont présentes
    
    Args:
        df: DataFrame à valider
        
    Returns:
        bool: True si valide, sinon lève une exception
    """
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    
    if missing:
        raise ValueError(f"Features manquantes: {missing}")
    
    return True
