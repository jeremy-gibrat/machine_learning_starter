"""
Configuration globale du projet ML
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Créer les dossiers s'ils n'existent pas
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.flightwatching.com")
ELASTIC_API_BASE_URL = os.getenv("ELASTIC_API_BASE_URL", "https://api.flightwatching.com")

# ============================================================================
# ML CONFIGURATION
# ============================================================================
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "app.log"))

# ============================================================================
# MODEL PARAMETERS (à personnaliser selon votre projet)
# ============================================================================
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_SEED
}

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Valide que la configuration est correcte"""
    errors = []
    
    if not API_KEY:
        errors.append("API_KEY non définie dans .env")
    
    if errors:
        raise ValueError(f"Erreurs de configuration: {', '.join(errors)}")
    
    return True

# Valider au chargement du module
try:
    validate_config()
except ValueError as e:
    print(f"⚠️ Attention: {e}")
