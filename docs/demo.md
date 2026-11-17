# Script de démonstration

Durée visée : 10 minutes. Docker Desktop doit être démarré avant de commencer.

## Préparation (avant la présentation)

Quatre fenêtres PowerShell, chacune dans `C:\Users\Doela\mlops-online-retail` avec `.venv\Scripts\Activate.ps1` :

- **A, API** : `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4`
- **B, MLflow** : `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- **C, commandes** : celle où l'on tape la démonstration
- **D, Docker** : `docker compose up -d`, puis `docker compose ps` (2 conteneurs actifs)

Onglets du navigateur : MLflow (http://127.0.0.1:5000), Prometheus (http://localhost:9090/targets, cible `api` UP), Grafana (http://localhost:3000, dossier MLOps, dashboards API et Pipeline), documentation de l'API (http://localhost:8000/docs).

Vérification : `python -m src.training.rollback` affiche un seul `champion`.

## Scénario 1 : fonctionnement nominal

1. Lancer le pipeline complet, dans la fenêtre C :

        python -m src.orchestration.flow

   Montrer les cinq étapes (ingestion, qualité, features, entraînement, rechargement de l'API) et la ligne `Version N | MAE=... -> PROMU`.

2. Dans MLflow, ouvrir l'expérience : paramètres, métriques, tags `git_commit` et `features_sha256`. Ouvrir la page du modèle : alias `champion` sur la nouvelle version.

3. Interroger l'API :

        $body = '{"qty_lag_1":120,"qty_lag_2":100,"qty_lag_4":90,"qty_roll_mean":105,"price_lag_1":1.25,"week_of_year":20,"month":5}'
        $r = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
        $r

   Montrer la version du modèle et la durée dans la réponse.

4. Envoyer un feedback, puis consulter les statistiques :

        Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/feedback -ContentType "application/json" -Body ('{"prediction_id":' + $r.prediction_id + ',"actual_quantity":260}')
        Invoke-RestMethod http://127.0.0.1:8000/stats

5. Une entrée invalide est refusée (code 422) et tracée :

        $bad = '{"qty_lag_1":-5,"qty_lag_2":100,"qty_lag_4":90,"qty_roll_mean":105,"price_lag_1":1.25,"week_of_year":20,"month":5}'
        try { Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $bad } catch { $_.Exception.Response.StatusCode.value__ }

6. Grafana : dashboard API (version du modèle, requêtes, latence, erreurs) et dashboard Pipeline (dernier pipeline à 1).

## Scénario 2 : incident contrôlé, lot de données invalide

1. Corrompre le lot de mars 2011 (colonne `Quantity` supprimée) :

        python -c "import pandas as pd; p='data/batches/batch_2011-03.parquet'; pd.read_parquet(p).drop(columns=['Quantity']).to_parquet(p, index=False); print('Lot corrompu')"

2. Relancer le pipeline sans réingérer :

        python -m src.orchestration.flow --skip-ingest

   Attendu : `quality_task` échoue avec `Colonnes manquantes : ['Quantity']`, ni les features ni l'entraînement ne sont exécutés.

3. Montrer l'alerte et l'absence de nouvelle version :

        Get-Content logs\alerts.log -Tail 2
        python -m src.training.rollback

4. Dashboard Pipeline : dernier pipeline à 0, exécutions `failure`.

5. Le service n'est pas interrompu : refaire une prédiction, la version servie est inchangée.

6. Réparer : `python -m src.orchestration.flow` (ingestion complète, lots régénérés). Le dashboard repasse à 1.

## Scénario 3 : modèle rejeté et retour arrière

1. Entraîner un modèle volontairement mauvais :

        python -m src.training.train n_estimators=1 max_depth=1

   Attendu : `-> REJETÉ`, le champion en place continue de servir.

2. Retour arrière vers une version précédente :

        python -m src.training.rollback
        python -m src.training.rollback 19

3. L'API prend en compte le changement en moins de 5 secondes par worker :

        Start-Sleep -Seconds 7
        1..20 | ForEach-Object { (Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body).model_version } | Group-Object | Select-Object Name, Count

   Attendu : une seule version, la 19.

## Scénario 4 (optionnel) : dépendance indisponible

Arrêter l'API (fermer la fenêtre A avec la croix), puis lancer `python -m src.orchestration.flow --skip-ingest`. Le pipeline se termine avec l'avertissement `API non rechargée` : l'indisponibilité de l'API ne bloque pas l'entraînement. Relancer l'API ensuite : elle charge le champion au démarrage.

## Questions probables

- Pourquoi autant de versions ? Les données n'ont pas changé, le même modèle est réentraîné et la règle de tolérance de 1 % le promeut.
- Pourquoi 4 workers ? Voir `docs/synthese_ops.md`, section 5 : environ 120 prédictions par seconde, p95 de 0,6 s à 50 utilisateurs.
- Pourquoi des échecs lors de la première campagne de charge ? Verrouillage de SQLite, corrigé par le mode WAL.
- Pourquoi un modèle sans identifiant de produit ? Voir les limites, section 7.
