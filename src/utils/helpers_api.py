"""
Fonctions utilitaires pour requêtes API HTTP et cURL
"""
import requests
import subprocess

def api_request(method, url, **kwargs):
    """
    Effectue une requête HTTP (GET, POST, etc.) via requests
    Args:
        method: 'get', 'post', 'put', 'delete', ...
        url: URL de l'API
        kwargs: params, data, json, headers, etc.
    Returns:
        requests.Response
    """
    method = method.lower()
    if not hasattr(requests, method):
        raise ValueError(f"Méthode HTTP non supportée: {method}")
    return getattr(requests, method)(url, **kwargs)

def curl_request(command):
    """
    Exécute une requête cURL en ligne de commande
    Args:
        command: str, la commande cURL complète (ex: 'curl -X GET https://api...')
    Returns:
        tuple (stdout, stderr)
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr
