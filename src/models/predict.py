"""
Module de prédiction
"""
import torch
import pandas as pd
from src.models.train import load_model
from src.models.train import load_pinn_model


def predict(X_new, model_filename):
    """
    Fait une prédiction avec le modèle entraîné
    
    Args:
        X_new (pd.DataFrame): Nouvelles données
        model_filename (str): Nom du fichier modèle
        
    Returns:
        predictions: Prédictions du modèle
    """
    # Charger le modèle
    model = load_model(model_filename)
    
    # Prédire
    predictions = model.predict(X_new)
    
    return predictions


def predict_proba(X_new, model_filename):
    """
    Fait une prédiction avec probabilités
    
    Args:
        X_new (pd.DataFrame): Nouvelles données
        model_filename (str): Nom du fichier modèle
        
    Returns:
        probabilities: Probabilités des classes
    """
    # Charger le modèle
    model = load_model(model_filename)
    
    # Prédire les probabilités
    probabilities = model.predict_proba(X_new)
    
    return probabilities

def predict_pinn(model, x, device="cpu"):
    """
    Prédit la sortie d'un PINN pour une entrée x
    """
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(x_tensor)
    return y_pred.cpu().numpy()

def predict_pinn_from_file(model_class, model_path, x, device="cpu", *args, **kwargs):
    """
    Charge un modèle PINN depuis un fichier et prédit la sortie pour x
    Args:
        model_class: classe du modèle (ex: PINN)
        model_path: chemin du fichier de poids
        x: entrée(s) pour la prédiction
        device: cpu ou cuda
        *args, **kwargs: paramètres pour instancier le modèle
    Returns:
        numpy array: prédiction
    """
    model = load_pinn_model(model_class, model_path, *args, **kwargs)
    return predict_pinn(model, x, device=device)