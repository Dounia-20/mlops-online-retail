# Prévision des quantités vendues - projet MLOps

Ce dépôt contient une chaîne MLOps complète construite autour d'un modèle de prévision des ventes : ingestion des données, contrôles qualité, entraînement suivi dans MLflow, API de prédiction, supervision et tests de charge. Tout s'exécute en local.

## Sujet

Le modèle prévoit la quantité vendue par produit et par semaine, pour les 50 produits les plus vendus d'un détaillant en ligne. Il s'agit d'un RandomForest volontairement simple, car le projet porte sur la chaîne d'exploitation et non sur la performance prédictive.

## Données

Les données viennent du dépôt UCI Machine Learning Repository, jeu de données Online Retail (n° 352, DOI 10.24432/C5BW33), créé par Daqing Chen (London South Bank University). Il contient 541 909 transactions d'un détaillant britannique, du 1er décembre 2010 au 9 décembre 2011. La licence est celle indiquée sur la page du jeu de données (CC BY 4.0), avec obligation de citer la source.

L'historique est rejoué par lots : il est découpé en 13 lots mensuels pour simuler l'arrivée de nouvelles données chaque mois.

Le fichier n'est pas versionné, car il est trop volumineux. Pour le récupérer, télécharger l'archive sur https://archive.ics.uci.edu/dataset/352/online+retail puis placer `Online Retail.xlsx` dans `data/raw/`.

## Architecture

    Online Retail.xlsx
          |
      ingestion  ->  13 lots mensuels (parquet)
          |
      contrôles qualité (bloquants)
          |
      features (produit x semaine)
          |
      entraînement + MLflow (registre, règle de mise en service)
          |
      alias "champion"  ->  API FastAPI  ->  SQLite (traces, feedbacks)
                                |
                            /metrics  ->  Prometheus  ->  Grafana

Le flux est orchestré avec Prefect. Les alertes d'échec sont écrites dans `logs/alerts.log`.

Organisation du code :

- `src/data` : ingestion et contrôles qualité
- `src/features` : construction des variables
- `src/training` : entraînement, mise en service et retour arrière
- `src/orchestration` : flux Prefect
- `src/api` : API, persistance SQLite et métriques
- `monitoring` : configuration de Prometheus et dashboards Grafana
- `tests` : tests unitaires et scénario de charge
- `scripts` : campagne de tests de charge et synthèse des résultats
- `docs` : preuves d'intégration, résultats de charge et synthèse

## Prérequis

Python 3.13, Git, PowerShell et Docker Desktop avec Docker Compose (à démarrer avant `docker compose up`). Un navigateur sert à consulter MLflow, Prometheus et Grafana.

## Installation

Le projet a été développé sous Windows avec Python 3.13.

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Pour retrouver exactement les versions utilisées pendant le développement, installer plutôt `requirements.lock` :

    pip install -r requirements.lock

Les paramètres (chemins, seuils de qualité, hyperparamètres, planification) sont dans `configs/config.yaml`.

## Utilisation

Premier lancement : exécuter le pipeline complet avant de démarrer l'API. Le registre MLflow (mlflow.db) n'est pas versionné, et l'API a besoin d'un modèle champion pour répondre.

Le pipeline complet enchaîne l'ingestion, les contrôles qualité, les features, l'entraînement et le rechargement de l'API :

    python -m src.orchestration.flow

Pour relancer sans réingérer les données :

    python -m src.orchestration.flow --skip-ingest

Pour un déclenchement automatique selon la planification définie dans la configuration (le premier de chaque mois à 6 h) :

    python -m src.orchestration.flow --serve

Chaque étape peut aussi être lancée séparément :

    python -m src.data.ingest
    python src/data/quality.py
    python -m src.features.build_features
    python -m src.training.train

Les hyperparamètres peuvent être modifiés en ligne de commande :

    python -m src.training.train n_estimators=300 max_depth=15

## Suivi des modèles

Chaque entraînement enregistre dans MLflow les paramètres, les métriques (MAE, RMSE, erreur d'une prévision naïve), le modèle, le commit Git et l'empreinte du fichier de features. Une nouvelle version devient le modèle en production (alias `champion`) si sa MAE est inférieure à celle de la prévision naïve et ne dépasse pas celle du champion de plus de 1 %. Sinon elle est marquée `rejected`.

Pour lister les versions ou revenir à une version précédente :

    python -m src.training.rollback
    python -m src.training.rollback 1

L'API prend en compte le changement de version : chaque worker vérifie l'alias `champion` toutes les 5 secondes, et `POST /reload` force le rechargement dans le worker qui le reçoit.

L'interface MLflow s'ouvre avec :

    mlflow ui --backend-store-uri sqlite:///mlflow.db

## API

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

Routes disponibles :

- `POST /predict` : prédiction, avec validation des entrées et version du modèle dans la réponse
- `POST /feedback` : quantité réellement vendue pour une prédiction donnée
- `GET /health` : état du service et version chargée
- `POST /reload` : rechargement du modèle champion
- `GET /stats` : nombre de prédictions et de feedbacks
- `GET /metrics` : métriques au format Prometheus

Exemple de corps pour `/predict` :

    {"qty_lag_1": 120, "qty_lag_2": 100, "qty_lag_4": 90, "qty_roll_mean": 105, "price_lag_1": 1.25, "week_of_year": 20, "month": 5}

Chaque requête est enregistrée dans `data/app.db` avec la version du modèle, la durée et l'éventuelle erreur.

## Supervision

Prometheus et Grafana se lancent avec Docker :

    docker compose up -d

Prometheus est accessible sur http://localhost:9090 et Grafana sur http://localhost:3000 (compte de démonstration admin / admin). Deux dashboards sont chargés automatiquement dans le dossier MLOps : un pour l'API (trafic, erreurs, latence, version du modèle) et un pour les exécutions du pipeline (statut, durée, ancienneté).

L'API tourne sur la machine hôte et Prometheus la interroge via `host.docker.internal:8000`.

## Tests

    pytest

Les 27 tests couvrent : contrôles qualité (`test_quality.py`), API (`test_api.py`), règle de promotion (dont la baseline) et retour arrière MLflow (`test_promotion.py`), enchaînement du flux Prefect, blocage sur lot invalide et alerte (`test_flow.py`).

## Tests de charge

    powershell -ExecutionPolicy Bypass -File scripts\run_load_tests.ps1
    python scripts\summarize_load.py

La campagne mesure l'API avec 1, 2 et 4 workers sous 10, 50 et 100 utilisateurs simultanés. Les résultats sont dans `docs/charge/resultats.md`.

## Synthèse

Les choix techniques, les preuves d'intégration, les résultats de charge, le dimensionnement recommandé et les limites connues sont détaillés dans `docs/synthese_ops.md`.


## Traitement des données

Lors de la construction des features, les lignes dont la quantité ou le prix est négatif ou nul (retours, annulations) sont exclues, ainsi que les codes qui ne sont pas des produits (frais de port, ajustements). Les ventes sont agrégées par produit et par semaine, avec les décalages de 1, 2 et 4 semaines et une moyenne glissante calculés uniquement sur le passé. Le jeu de test est formé des 8 dernières semaines.

## Règles de mise en service

La décision repose sur la MAE mesurée sur les 8 dernières semaines. Au premier entraînement, aucun champion n'existe : la version est promue si elle bat la prévision naïve (MAE strictement inférieure à celle de la semaine précédente). Avec un champion, elle doit aussi battre la prévision naïve, puis elle est promue si sa MAE ne dépasse pas celle du champion de plus de 1 %. Dans le cas contraire elle est marquée `rejected` et le champion en place continue de servir. Le retour arrière remet l'alias `champion` sur une version déjà enregistrée dans MLflow, sauf si cette version a été rejetée : il est alors refusé.

## Documentation de l'API

La documentation interactive est disponible sur http://localhost:8000/docs.

Réponse de `/predict` :

    {"prediction_id": 1, "prediction": 274.38, "model_version": "11", "duration_ms": 57.13}

Corps de `/feedback` :

    {"prediction_id": 1, "actual_quantity": 260, "comment": "optionnel"}

## Indicateurs supervisés

- API : requêtes par code HTTP, erreurs, latence p50 et p95.
- Modèle : version active.
- Utilisation : prédictions par version, feedbacks (via `/stats`).
- Pipeline : statut, durée et ancienneté de la dernière exécution, décompte par statut.

La consommation de processeur et de mémoire n'est pas supervisée : elle a été mesurée ponctuellement pendant les tests de charge.

## Démonstrations

Fonctionnement nominal : lancer `python -m src.orchestration.flow`, constater dans MLflow la nouvelle version et sa décision, envoyer une prédiction à l'API, puis consulter les dashboards Grafana.

Incident contrôlé : retirer la colonne `Quantity` du lot `data/batches/batch_2011-03.parquet`, relancer `python -m src.orchestration.flow --skip-ingest`. Le contrôle qualité bloque le pipeline, aucune version n'est créée, une alerte est écrite dans `logs/alerts.log` et le dashboard Pipeline passe à 0. Une relance complète régénère les lots et rétablit la situation. Les traces sont dans `docs/preuves`.

## Résultats de charge

Mesures sur la route `/predict`, après correction du verrouillage de SQLite, avec 12 processeurs logiques disponibles.

| Workers | Utilisateurs | Débit (req/s) | p95 (ms) | Erreurs |
|---:|---:|---:|---:|---:|
| 1 | 50 | 47,0 | 1 200 | 0 |
| 2 | 50 | 83,2 | 870 | 0 |
| 4 | 50 | 125,8 | 600 | 0 |
| 4 | 100 | 117,8 | 2 600 | 0 |

Dimensionnement recommandé : 4 workers, 4 vCPU et 2 Go de mémoire, pour une charge nominale d'environ 85 prédictions par seconde. Le détail est dans `docs/synthese_ops.md`.

## Déclenchement planifié

Le déclenchement automatique demande un serveur Prefect local, à lancer dans une fenêtre dédiée :

    prefect server start

Dans une autre fenêtre :

    $env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
    python -m src.orchestration.flow --serve

La planification par défaut est mensuelle (le premier jour du mois à 6 h). Pour tester avec une fréquence courte : `--cron "*/2 * * * *"`. Le processus doit rester actif pour que les exécutions soient déclenchées.


Pour la supervision, lancer l'API avec un seul processus (`uvicorn src.api.main:app --host 0.0.0.0 --port 8000`) : avec plusieurs workers, les métriques Prometheus sont incohérentes. Le mode multi-workers sert aux tests de charge et au dimensionnement.
