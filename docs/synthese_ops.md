# Synthèse des choix et résultats OPS

## 1. Périmètre

Le projet met en place une chaîne MLOps autour d'un modèle de prévision des quantités vendues par produit et par semaine (50 produits, jeu de données UCI Online Retail). Le modèle est un RandomForest volontairement simple. Sa MAE est de 520,8 contre 548,2 pour une prévision naïve (quantité de la semaine précédente), ce qui suffit pour illustrer les décisions de mise en service. Le RMSE est beaucoup plus élevé (4 104) à cause de quelques pics de ventes.

## 2. Décisions techniques

| Besoin | Choix | Justification |
|---|---|---|
| Orchestration | Prefect | S'exécute en local sans infrastructure, gère les tentatives, les hooks d'échec et la planification par cron. |
| Suivi des modèles | MLflow avec base SQLite | Registre de versions, alias et tags sans serveur dédié. |
| Format des données | Parquet | Lecture rapide, types conservés. La lecture du fichier xlsx d'origine prend une trentaine de secondes. |
| API | FastAPI | Validation des entrées par schéma, documentation automatique. |
| Persistance | SQLite en mode WAL | Aucun service à installer. Limite connue, voir section 6. |
| Supervision | Prometheus et Grafana, dashboards chargés par fichiers | Les dashboards sont versionnés dans le dépôt et se rechargent à l'identique. |
| Charge | Locust | Scénario écrit en Python, sorties CSV exploitables. |

Les paramètres (chemins, seuils de qualité, hyperparamètres, tolérance de promotion, planification) sont séparés du code dans `configs/config.yaml`.

## 3. Intégration de bout en bout

### 3.1 Fonctionnement nominal

Un lancement de `python -m src.orchestration.flow` enchaîne cinq étapes : ingestion (13 lots mensuels), contrôle qualité des 13 lots, construction des features (2 500 lignes, 10 colonnes), entraînement avec enregistrement dans MLflow, puis rechargement de l'API. Lors d'un essai nominal, une nouvelle version a été promue et l'API est passée de la version 11 à la version 12 sans intervention. Le dashboard API l'affiche (`docs/preuves/grafana_api.png`).

### 3.2 Incident contrôlé : lot invalide

Le lot de mars 2011 a été volontairement altéré (colonne `Quantity` supprimée), puis le pipeline relancé.

- Le contrôle qualité a levé `Colonnes manquantes : ['Quantity']` et le pipeline s'est arrêté à cette étape.
- Ni les features ni l'entraînement n'ont été exécutés : aucune nouvelle version du modèle n'a été créée.
- Une alerte a été écrite dans `logs/alerts.log` (copie : `docs/preuves/alerts.log`) et la trace de l'exécution est conservée dans `docs/preuves/incident_lot_invalide.txt`.
- Le dashboard Pipeline est passé à 0 (dernier pipeline en échec), avec une durée de 0,5 s correspondant à l'arrêt immédiat (`docs/preuves/grafana_pipeline_incident.png`).
- La relance complète a régénéré les lots sains et remis le dashboard à 1.

Le contrôle porte sur la présence des colonnes, un minimum de 100 lignes par lot, l'absence de valeurs nulles dans les colonnes clés, un taux de description manquante inférieur à 1 % et les types numériques et date. Cinq tests automatisés vérifient ces règles.

### 3.3 Règle de mise en service et retour arrière

Une nouvelle version devient `champion` si sa MAE est inférieure à celle de la prévision naïve et ne dépasse pas celle du champion de plus de 1 %. Sinon elle est marquée `rejected`. Le retour arrière est refusé vers une version rejetée.

- Un modèle volontairement dégradé (1 arbre, profondeur 1) a donné une MAE de 552,5 contre 520,8 : il a été rejeté et la version 1 est restée en production.
- Avec une tolérance de 0 %, plusieurs versions quasi équivalentes (MAE entre 520,8 et 524,7) étaient rejetées. La tolérance a été portée à 1 % car, en dessous de cet écart, la différence relève du bruit.
- Une version promue (la version 8) a ensuite été remplacée par un retour arrière vers la version 1 (`python -m src.training.rollback 1`) : la version 1 est redevenue champion et la version 8 a été archivée. La liste des versions et de leurs statuts est conservée dans MLflow (`docs/preuves/mlflow_registre.png`).

Chaque exécution enregistre aussi les paramètres, les métriques, le commit Git et l'empreinte du fichier de features.

### 3.4 API, traces et feedbacks

L'API valide les entrées (une quantité négative renvoie une erreur 422, tracée en base), renvoie la version du modèle utilisée et la durée de chaque prédiction, et enregistre chaque appel dans SQLite. Un feedback sur une prédiction inconnue renvoie 404. Le décompte des prédictions et des feedbacks est disponible sur `/stats`.

### 3.5 Supervision

Deux dashboards sont chargés automatiquement dans Grafana. Le premier suit le trafic par code HTTP, la latence p50 et p95, les erreurs, le nombre de prédictions et la version du modèle. Le second suit le statut du dernier pipeline, sa durée, son ancienneté et le décompte des exécutions par statut.

## 4. Tests de charge

### 4.1 Protocole

Le scénario Locust envoie 90 % de prédictions et 10 % de feedbacks ou de contrôles de santé, avec 0,05 à 0,2 s d'attente par utilisateur. Chaque combinaison de 1, 2 et 4 workers et de 10, 50 et 100 utilisateurs simultanés est mesurée pendant 30 secondes. Les valeurs ci-dessous concernent la route `/predict`. La machine de test dispose de 12 processeurs logiques.

### 4.2 Première campagne

Des erreurs HTTP 500 sont apparues dès 50 utilisateurs : jusqu'à 2,2 % avec 4 workers à 50 utilisateurs et 9,9 % avec 2 workers à 100 utilisateurs, avec des p99 proches de 6 secondes. Les erreurs touchaient `/predict` et `/feedback`, deux routes qui écrivent dans SQLite. Le délai de 6 secondes est cohérent avec le délai d'attente par défaut de SQLite (5 secondes) sur une base verrouillée. L'exception exacte n'a pas été relevée : la cause est une hypothèse, confirmée par le résultat de la correction.

### 4.3 Correction et seconde campagne

La base a été passée en mode WAL, avec un délai d'attente de 30 secondes. Les 9 mesures ne comportent plus aucune erreur.

| Workers | Utilisateurs | Prédictions/s | Médiane (ms) | p95 (ms) | p99 (ms) | Échecs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 41,5 | 90 | 160 | 240 | 0 |
| 1 | 50 | 47,0 | 790 | 1 200 | 1 600 | 0 |
| 1 | 100 | 45,0 | 1 800 | 2 400 | 3 200 | 0 |
| 2 | 10 | 41,2 | 92 | 160 | 240 | 0 |
| 2 | 50 | 83,2 | 360 | 870 | 1 900 | 0 |
| 2 | 100 | 82,5 | 500 | 3 300 | 8 800 | 0 |
| 4 | 10 | 51,4 | 44 | 69 | 94 | 0 |
| 4 | 50 | 125,8 | 150 | 600 | 2 100 | 0 |
| 4 | 100 | 117,8 | 220 | 2 600 | 6 600 | 0 |

Comparaison avant et après correction, pour les cas les plus touchés :

| Cas | Avant : débit, échecs, p95 | Après : débit, échecs, p95 |
|---|---|---|
| 2 workers, 100 utilisateurs | 62,5 /s, 9,9 %, 6,3 s | 82,5 /s, 0 %, 3,3 s |
| 4 workers, 50 utilisateurs | 79,3 /s, 2,2 %, 2,4 s | 125,8 /s, 0 %, 0,6 s |
| 4 workers, 100 utilisateurs | 81,7 /s, 8,6 %, 6,0 s | 117,8 /s, 0 %, 2,6 s |

Les résultats complets de chaque campagne sont dans `docs/charge_avant` et `docs/charge`.

### 4.4 Lecture

- Un worker plafonne autour de 45 prédictions par seconde.
- Le débit progresse presque proportionnellement avec le nombre de workers : à 50 utilisateurs, 47, 83 puis 126 prédictions par seconde pour 1, 2 et 4 workers.
- À 100 utilisateurs, 4 workers saturent : le débit ne progresse plus (118 par seconde) et le p95 monte à 2,6 s, sans erreur.
- Chaque worker occupe environ 290 Mo de mémoire (mesure faite avec 4 workers : 4 processus de 287 à 291 Mo, plus 35 Mo pour les processus de lancement).

## 5. Dimensionnement recommandé

Objectif retenu : p95 inférieur à 1 seconde, sans erreur.

| Configuration | Capacité observée | Verdict |
|---|---|---|
| 1 worker | environ 40 prédictions/s | Insuffisant dès 50 utilisateurs (p95 de 1,2 s) |
| 2 workers | environ 80 prédictions/s | À la limite (p95 de 0,87 s à 50 utilisateurs) |
| 4 workers | environ 120 prédictions/s | Retenu (p95 de 0,6 s à 50 utilisateurs) |

Recommandation : 4 workers, soit 4 vCPU et 2 Go de mémoire (1,15 Go pour les workers, le reste en marge). La capacité maximale mesurée est d'environ 120 prédictions par seconde. En gardant 30 % de marge, la charge nominale visée est d'environ 85 prédictions par seconde. Au-delà, il faudrait ajouter des instances derrière un répartiteur de charge et remplacer SQLite.

## 6. Limites connues

- Locust et l'API tournent sur la même machine : les chiffres sont des ordres de grandeur, et les deux se disputent le processeur.
- Chaque combinaison n'a été mesurée qu'une fois pendant 30 secondes. L'écart observé entre 42 et 51 prédictions par seconde pour une même charge à 10 utilisateurs donne une idée de la variabilité.
- SQLite reste le goulot d'étranglement en cas de montée en charge : les écritures sont sérialisées. Une base client-serveur serait nécessaire en production.
- Le scénario est plus agressif qu'un usage réel (attente de 0,05 à 0,2 s entre requêtes) : le nombre d'utilisateurs réels supportés est probablement supérieur.
- L'API reçoit les variables déjà calculées (décalages, moyenne glissante) : elle ne recalcule pas les features à partir de l'historique des ventes.
- Les feedbacks sont stockés mais ne servent ni à surveiller la dérive du modèle ni à réentraîner automatiquement.
- Le déclenchement planifié a été démontré avec une fréquence de 2 minutes (voir section 8), mais pas sur plusieurs mois : la planification de production (le premier de chaque mois à 6 h) n'a pas été observée.
- L'API ne tourne pas dans un conteneur : seuls Prometheus et Grafana sont conteneurisés, et Prometheus joint l'API par `host.docker.internal`, adapté à Docker Desktop sous Windows.
- Les identifiants Grafana (admin / admin) sont ceux d'une démonstration locale et ne conviennent pas à un déploiement.
- Les tests automatisés (27 au total) couvrent la qualité des données, l'API, la règle de promotion, le retour arrière et l'enchaînement du flux Prefect (blocage avant entraînement sur lot invalide, alerte). L'exécution réelle de Prefect avec MLflow et l'API reste validée par les scénarios d'intégration décrits plus haut.
- La supervision de l'API est fiable avec un seul processus. Avec plusieurs workers, chacun tient ses propres compteurs et Prometheus interroge l'un d'eux au hasard : les courbes deviennent incohérentes. Le mode multiprocessus du client Prometheus n'est pas configuré. Les mesures de charge, issues de Locust, ne sont pas concernées. Les captures du dashboard API ont été faites avec un seul worker.
- Le modèle ne couvre que 50 produits et ne gère pas les produits nouveaux ou rares.

## 7. Correctif après relecture : rechargement avec plusieurs workers

Le rechargement du modèle par `POST /reload` n'atteignait qu'un seul des workers : avec 4 workers, les autres continuaient à servir l'ancienne version. Ce défaut n'était pas visible pendant les tests de charge, car la version du modèle n'y changeait pas. Chaque worker vérifie désormais l'alias `champion` toutes les 5 secondes et recharge le modèle si la version a changé. Cette vérification rend aussi l'API tolérante à une indisponibilité du registre : elle garde le modèle déjà chargé.

Le test a consisté à lancer un pipeline sur une API à 4 workers (version 14 promue), puis à interroger `/predict` 40 fois : les 40 réponses indiquent la version 14.

Deux détails de la préparation des données restent à signaler : les 50 produits sont choisis sur l'ensemble de la période, et les prix manquants sont comblés avec le prix de la semaine suivante. Leur effet sur la MAE est probablement faible, mais il n'a pas été mesuré. Le modèle est enfin global et ne reçoit pas d'identifiant produit : il apprend des échelles de ventes via les décalages, pas des comportements propres à chaque produit.


## 8. Déclenchement automatique

Le flux est servi avec `python -m src.orchestration.flow --serve`, relié à un serveur Prefect local (`prefect server start`). Un serveur temporaire, celui que Prefect lance par défaut, ne sait pas planifier : un premier essai l'a montré par l'avertissement `Cannot schedule flows on an ephemeral server`.

Le test a été fait avec la fréquence `*/2 * * * *` (option `--cron`) : sans intervention, le pipeline s'est déclenché toutes les 2 minutes, a entraîné une nouvelle version (jusqu'à la version 20), l'a promue puis a rechargé l'API. Les données n'ayant pas changé, chaque version obtient la même MAE et la règle de tolérance la promeut. Chaque déclenchement relance aussi l'ingestion, soit environ 30 secondes de lecture du fichier xlsx. La fréquence configurée par défaut est mensuelle (le premier jour du mois à 6 h).

