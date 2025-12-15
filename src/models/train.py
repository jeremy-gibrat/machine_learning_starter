from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

"""
Module d'entraînement du modèle
"""

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config.config import MODEL_DIR, RANDOM_SEED, TEST_SIZE, MODEL_PARAMS


def train_model_classifier(X, y, model_params=None):
    """
    Entraîne un modèle Random Forest
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        model_params (dict): Hyperparamètres du modèle
        
    Returns:
        model: Modèle entraîné
        metrics: Métriques d'évaluation
    """
    # Utiliser les paramètres par défaut si non fournis
    if model_params is None:
        model_params = MODEL_PARAMS
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    print(f"📊 Train set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Entraînement
    print("\n🚀 Entraînement du modèle...")
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné")
    
    # Évaluation
    print("\n📈 Évaluation...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }
    
    return model, metrics

def train_model_regressor(X, y, model_params=None):
    """
    Entraîne un modèle de régression Random Forest
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Cible
        model_params (dict): Hyperparamètres du modèle
    Returns:
        model: Modèle entraîné
        metrics: Métriques d'évaluation
    """
    if model_params is None:
        model_params = MODEL_PARAMS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"📊 Train set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    print("\n🚀 Entraînement du modèle de régression...")
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné")
    print("\n📈 Évaluation...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    metrics = {
        'mse': mse,
        'r2': r2
    }
    return model, metrics

def save_model(model, filename):
    """
    Sauvegarde le modèle entraîné
    
    Args:
        model: Modèle à sauvegarder
        filename (str): Nom du fichier
    """
    filepath = MODEL_DIR / filename
    joblib.dump(model, filepath)
    print(f"✅ Modèle sauvegardé: {filepath}")


def load_model(filename):
    """
    Charge un modèle sauvegardé
    
    Args:
        filename (str): Nom du fichier
        
    Returns:
        model: Modèle chargé
    """
    filepath = MODEL_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✅ Modèle chargé: {filepath}")
    return model



class PINN(nn.Module):
    """
    Physics-Informed Neural Network de base
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def train_pinn(model, optimizer, loss_fn, data_loader, n_epochs=1000, device="cpu"):
        """
        Entraîne un PINN sur les données fournies
        """
        model.to(device)
        model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} - Loss: {total_loss/len(data_loader):.6f}")
        return model

    def save_pinn_model(model, path):
        """
        Sauvegarde les poids du modèle PINN
        """
        import torch
        torch.save(model.state_dict(), path)

    def load_pinn_model(model_class, path, *args, **kwargs):
        """
        Charge les poids dans une instance de PINN
        Args:
            model_class: classe du modèle (ex: PINN)
            path: chemin du fichier de poids
            *args, **kwargs: paramètres pour instancier le modèle
        Returns:
            modèle PINN avec poids chargés
        """
        import torch
        model = model_class(*args, **kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model