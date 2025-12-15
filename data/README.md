# Données

Ce dossier contient les données du projet ML.

## Structure

- `raw/` : Données brutes, non modifiées
- `processed/` : Données nettoyées et transformées, prêtes pour le ML

## ⚠️ Important

Les données ne sont **PAS versionnées dans Git** pour des raisons de :
- Taille (éviter de surcharger le repository)
- Confidentialité (données sensibles)
- Performance (Git n'est pas fait pour les gros fichiers binaires)

## Format recommandé

- CSV pour les données tabulaires
- Parquet pour les gros volumes (plus performant)
- JSON pour les données structurées

## Nomenclature

```
dataset_name_YYYYMMDD.csv
```

Exemple : `apu_data_20251112.csv`
