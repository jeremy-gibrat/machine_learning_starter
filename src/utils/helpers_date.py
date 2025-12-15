"""
Fonctions utilitaires pour les dates
"""
from datetime import datetime, timezone
import pandas as pd

def is_valid_date(date_str, fmt="%Y-%m-%d"):
    """
    Vérifie si une chaîne est une date valide selon un format.
    """
    try:
        datetime.strptime(date_str, fmt)
        return True
    except (ValueError, TypeError):
        return False

def format_date(date_obj, fmt="%Y-%m-%d"):
    """
    Formate un objet date selon un format donné.
    """
    try:
        return date_obj.strftime(fmt)
    except Exception:
        return None

def to_utc(date_obj):
    """
    Convertit un datetime en UTC.
    """
    try:
        return date_obj.astimezone(timezone.utc)
    except Exception:
        return None

def to_locale(date_obj, tz=None):
    """
    Convertit un datetime en timezone locale (ou spécifiée).
    """
    try:
        if tz:
            return date_obj.astimezone(tz)
        return date_obj.astimezone()
    except Exception:
        return None

# Parse a date and a time into a datetime object, handles multiple formats
# If to_utc is True, convert from the given timezone to UTC
def parse_datetime(date_str, time_str, to_utc=False, timezone_name=None, dayfirst=False):
    """
    Parse une date et une heure en datetime, gère plusieurs formats et conversion UTC.
    """

    try:
        # Nettoyage time_str
        ts = time_str
        if ts is None or (isinstance(ts, float) and pd.isnull(ts)):
            return pd.NaT
        ts = str(int(ts)) if isinstance(ts, float) and ts == int(ts) else str(ts)
        # Parsing date
        date = pd.to_datetime(str(date_str), errors='coerce', dayfirst=dayfirst)
        if pd.isnull(date):
            date = pd.to_datetime(str(date_str), errors='coerce', dayfirst=not dayfirst)
        if pd.isnull(date):
            return pd.NaT
        # Parsing heure
        result = parse_time_str(ts)
        if result is None:
            return pd.NaT
        hour, minute = result
        dt = datetime(date.year, date.month, date.day, hour, minute)
        if to_utc and timezone_name:
            try:
                import pytz
                local = pytz.timezone(timezone_name)
                dt = local.localize(dt).astimezone(pytz.utc)
            except Exception:
                return pd.NaT
        return dt
    except Exception:
        return pd.NaT

def parse_time_str(time_str):
    """
    Parse une chaîne d'heure (formats : 12:39:00 AM, 8:16, 1446, etc.)
    Retourne (hour, minute) ou None si non reconnu.
    """
    import re
    s = str(time_str).strip()
    # Format 12:39:00 AM
    try:
        t = datetime.strptime(s, '%I:%M:%S %p')
        return t.hour, t.minute
    except Exception:
        pass
    # Format HH:MM ou H:MM
    try:
        t = datetime.strptime(s, '%H:%M')
        return t.hour, t.minute
    except Exception:
        pass
    # Format HHMM ou HMM (e.g. 1446, 816)
    s_digits = re.sub(r'\D', '', s)
    if s_digits.isdigit():
        if len(s_digits) == 4:
            hour = int(s_digits[:2])
            minute = int(s_digits[2:])
        elif len(s_digits) == 3:
            hour = int(s_digits[0])
            minute = int(s_digits[1:])
        else:
            return None
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour, minute
    return None