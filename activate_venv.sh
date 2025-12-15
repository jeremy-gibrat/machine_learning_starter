#!/bin/bash

# ============================================================================
# ACTIVATION RAPIDE DE L'ENVIRONNEMENT VIRTUEL
# ============================================================================
# Usage: source activate_venv.sh

VENV_NAME="venv"

if [ -d "$VENV_NAME" ]; then
    source "$VENV_NAME/bin/activate"
    echo "✅ Environnement virtuel activé"
    echo "Python: $(python --version)"
    echo "Pip: $(pip --version)"
else
    echo "❌ Environnement virtuel non trouvé"
    echo "Exécutez d'abord: bash setup_venv.sh"
fi
