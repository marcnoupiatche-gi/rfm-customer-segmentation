# Application Streamlit — Segmentation Client RFM

## Contenu
- `app.py` — code de l'application Streamlit
- `requirements.txt` — dépendances Python
- `scaler.pkl`, `kmeans.pkl`, `labels_map.pkl` — modèle entraîné (générés par le notebook d'analyse)
- `rfm_clustered.csv` — données RFM des 4 338 clients avec leur segment

## Installation et lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur (par défaut sur http://localhost:8501).

## Fonctionnalités
1. **Vue d'ensemble** : métriques clés, répartition des clients et du chiffre d'affaires par segment.
2. **Exploration des segments** : nuage de points interactif, distributions par variable, tableau filtrable.
3. **Prédire un segment** : saisie de Recency / Frequency / Monetary pour un client, prédiction instantanée de son segment, recommandations marketing associées et positionnement visuel par rapport à la base existante.

## Important
Les fichiers `.pkl` et le `.csv` doivent rester dans le **même dossier** que `app.py` pour que l'application fonctionne. Si vous souhaitez ré-entraîner le modèle sur de nouvelles données, relancez le notebook d'analyse (`RFM_Clustering_Notebook.ipynb`), qui régénère automatiquement ces fichiers.
