#!/bin/bash

# ============================================================================
# SETUP ENVIRONNEMENT VIRTUEL - TEMPLATE ML
# ============================================================================
# Ce script crée l'environnement virtuel Python et installe les dépendances
# Usage: bash setup_venv.sh

echo "=========================================="
echo "🚀 Setup Environnement Virtuel"
echo "=========================================="
echo ""

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Vérifier que Python 3 est installé
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 n'est pas installé${NC}"
    echo "Installez Python 3.9+ puis relancez ce script"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✅ Python trouvé: ${PYTHON_VERSION}${NC}"
echo ""

# Nom de l'environnement virtuel
VENV_NAME="venv"

# Supprimer l'ancien venv s'il existe
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}⚠️  Environnement virtuel existant détecté${NC}"
    read -p "Voulez-vous le supprimer et le recréer? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
        echo -e "${GREEN}✅ Ancien environnement supprimé${NC}"
    else
        echo -e "${YELLOW}⚠️  Utilisation de l'environnement existant${NC}"
    fi
fi

# Créer le venv s'il n'existe pas
if [ ! -d "$VENV_NAME" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv "$VENV_NAME"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Environnement virtuel créé${NC}"
    else
        echo -e "${RED}❌ Erreur lors de la création du venv${NC}"
        exit 1
    fi
fi
echo ""

# Activer le venv
echo "🔧 Activation de l'environnement virtuel..."
source "$VENV_NAME/bin/activate"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Environnement activé${NC}"
else
    echo -e "${RED}❌ Erreur lors de l'activation${NC}"
    exit 1
fi
echo ""

# Mettre à jour pip
echo "📦 Mise à jour de pip..."
pip install --upgrade pip
echo ""

# Installer les dépendances
if [ -f "requirements.txt" ]; then
    echo "📦 Installation des dépendances depuis requirements.txt..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Dépendances installées${NC}"
    else
        echo -e "${RED}❌ Erreur lors de l'installation des dépendances${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  Fichier requirements.txt non trouvé${NC}"
fi
echo ""

# Créer le fichier .env s'il n'existe pas
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "📝 Création du fichier .env..."
    cp .env.example .env
    echo -e "${GREEN}✅ Fichier .env créé depuis .env.example${NC}"
    echo -e "${YELLOW}⚠️  N'oubliez pas de le configurer avec vos valeurs!${NC}"
fi
echo ""

# Résumé
echo "=========================================="
echo -e "${GREEN}✅ INSTALLATION TERMINÉE${NC}"
echo "=========================================="
echo ""
echo "📋 Prochaines étapes:"
echo ""
echo "1. Activer l'environnement:"
echo "   source activate_venv.sh"
echo ""
echo "2. Configurer .env avec vos variables:"
echo "   nano .env"
echo ""
echo "3. Tester l'installation:"
echo "   python -c \"import pandas, numpy, sklearn; print('✅ OK')\""
echo ""
echo "4. Lancer Jupyter (optionnel):"
echo "   jupyter notebook"
echo ""
echo "=========================================="
