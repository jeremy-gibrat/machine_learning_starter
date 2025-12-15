"""
Fonctions utilitaires pour requêtes API HTTP et cURL
"""
from config.config import API_KEY, API_BASE_URL, ELASTIC_API_BASE_URL
import requests
import json
import subprocess

def api_request(method, endpoint, api_key=None, base_url=None, **kwargs):
    """
    Effectue une requête HTTP (GET, POST, etc.) via requests, avec gestion API_KEY et BASE_URL
    Args:
        method: 'get', 'post', 'put', 'delete', ...
        endpoint: chemin relatif ou URL complète
        api_key: clé API à utiliser (défaut: config)
        base_url: base URL à utiliser (défaut: config)
        kwargs: params, data, json, headers, etc.
    Returns:
        requests.Response
    """
    method = method.lower()
    if not hasattr(requests, method):
        raise ValueError(f"Méthode HTTP non supportée: {method}")
    if base_url is None:
        base_url = API_BASE_URL
    url = endpoint if endpoint.startswith("http") else f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = kwargs.pop("headers", {})
    if api_key is None:
        api_key = API_KEY
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    try:
        response = getattr(requests, method)(url, headers=headers, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return None

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

def elastic_search_request(payload: dict, api_key=None, base_url=None, endpoint="/_search") -> dict:
    """
    Effectue une requête POST à l'API ElasticSearch avec la clé API et l'URL du config par défaut.
    Retourne la réponse JSON sous forme de dictionnaire.
    """
    if api_key is None:
        api_key = API_KEY
    if base_url is None:
        base_url = ELASTIC_API_BASE_URL
    url = base_url.rstrip("/") + endpoint if endpoint.startswith("/") else base_url.rstrip("/") + "/" + endpoint
    headers = {
        "Authorization": f"ApiKey {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"[ERROR] ElasticSearch request failed: {e}")
        return None