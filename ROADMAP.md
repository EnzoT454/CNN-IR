# Feuille de route — Rendre CNN-IR utilisable sur de nouveaux spectres

## Objectif

L'objectif est de transformer ce dépôt de recherche en un outil capable de
recevoir un ou plusieurs spectres IR numériques et de retourner, de manière
reproductible, la probabilité de présence des 37 groupes fonctionnels du modèle.

L'expérience utilisateur minimale visée est la suivante :

```bash
cnn-ir predict \
  --input mon_spectre.csv \
  --y-unit absorbance \
  --output predictions.json
```

Exemple de résultat :

```json
{
  "sample_id": "mon_spectre",
  "model_version": "cnn-ir-original-37fg",
  "threshold": 0.5,
  "predictions": [
    {"group": "Alkane", "probability": 0.98, "present": true},
    {"group": "Arene", "probability": 0.91, "present": true},
    {"group": "Ketone", "probability": 0.76, "present": true}
  ],
  "warnings": []
}
```

Ce résultat est une **prédiction de groupes fonctionnels**, pas une
identification exacte de la molécule, une attribution des pics ni une preuve de
réussite d'une réaction.

## État initial

Le dépôt contient l'architecture, les traitements expérimentaux et les scripts
d'évaluation de l'article, mais il ne fournit pas encore une application
d'inférence utilisable :

- aucun modèle entraîné n'est présent ;
- les données d'entraînement et les scripts de téléchargement sont absents ;
- le prétraitement est couplé à la construction complète des datasets NIST et
  SDBS ;
- il n'existe pas de commande pour charger un CSV utilisateur ;
- les chemins des modèles sont incohérents entre entraînement et évaluation ;
- la pile Python 3.7/Keras/PlaidML est ancienne ;
- aucune vérification automatique ne garantit qu'un spectre utilisateur est
  compatible avec les données vues pendant l'entraînement.

La priorité est donc de créer un pipeline d'inférence isolé. La reconstruction
complète de l'étude pourra rester un chantier séparé.

## Contrat d'entrée cible

### Format minimal

Le premier format pris en charge sera un CSV avec une ligne d'en-tête et deux
colonnes numériques :

```csv
wavenumber,intensity
4000.0,0.982
3998.0,0.981
3996.0,0.979
400.0,0.751
```

Le nom des colonnes devra pouvoir être configuré en ligne de commande.

### Unités acceptées

L'axe spectral sera exprimé en nombres d'onde (`cm-1`). Les intensités pourront
être fournies sous trois formes explicites :

| Valeur de `--y-unit` | Données attendues | Conversion |
|---|---|---|
| `transmittance_fraction` | transmittance entre 0 et 1 | aucune |
| `percent_transmittance` | transmittance entre 0 et 100 | division par 100 |
| `absorbance` | absorbance | `T = 10 ** (-A)` |

L'unité ne devra jamais être devinée silencieusement. Une détection automatique
pourra seulement émettre une suggestion ou un avertissement.

### Représentation fournie au modèle

Après validation et conversion, chaque spectre sera interpolé sur :

```python
np.linspace(4000.0, 400.0, 600)
```

La forme finale sera :

- `(1, 600, 1)` pour un spectre ;
- `(nombre_de_spectres, 600, 1)` pour un lot.

### Métadonnées recommandées

Les métadonnées ne seront pas données au CNN, mais devront être conservées dans
le rapport d'inférence lorsqu'elles sont disponibles :

- identifiant de l'échantillon ;
- mode de mesure, par exemple ATR, transmission ou KBr ;
- état physique ;
- instrument ;
- résolution en cm⁻¹ ;
- unité d'intensité originale ;
- date et éventuels traitements instrumentaux.

Ces informations sont nécessaires pour analyser un éventuel décalage entre les
données utilisateur et le domaine d'entraînement.

## Architecture cible du dépôt

La logique réutilisable devra être déplacée dans un package, tandis que les
anciens scripts seront conservés temporairement pour la reproductibilité de
l'étude.

```text
CNN-IR/
├── README.md
├── ROADMAP.md
├── pyproject.toml
├── src/
│   └── cnn_ir/
│       ├── __init__.py
│       ├── cli.py              # commande cnn-ir
│       ├── config.py           # axe spectral et paramètres communs
│       ├── labels.py           # ordre canonique des 37 classes
│       ├── preprocessing.py    # validation et préparation d'un spectre
│       ├── model.py            # chargement/version du modèle
│       ├── inference.py        # prédiction et application des seuils
│       └── reporting.py        # sorties JSON et CSV
├── tests/
│   ├── fixtures/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_cli.py
├── examples/
│   ├── spectrum.csv
│   └── metadata.json
├── models/                     # ignoré par Git
│   ├── README.md               # origine et procédure de téléchargement
│   └── manifest.json           # version, SHA-256, labels, framework
└── scripts/                    # scripts historiques de l'article
```

## Phase 0 — Figer le périmètre et les données utilisateur

### Travail

1. Recenser les formats réels à tester : CSV exporté par l'instrument, JCAMP-DX
   ou autre format propriétaire.
2. Confirmer les unités x et y de chaque export.
3. Documenter le mode de mesure, la résolution et la couverture spectrale.
4. Sélectionner quelques spectres représentatifs, idéalement accompagnés de
   molécules ou groupes fonctionnels connus.
5. Définir le résultat attendu : probabilités complètes, classes au-dessus d'un
   seuil, comparaison de plusieurs échantillons ou traitement par lot.

### Livrable

Un petit jeu d'acceptation non sensible comprenant au minimum :

- un spectre en transmittance ;
- un spectre en absorbance ;
- les métadonnées associées ;
- les groupes fonctionnels attendus lorsque ceux-ci sont connus.

### Critère de fin

Les formats, unités et cas d'utilisation ne sont plus ambigus. Aucun code
d'inférence ne doit être écrit sur la base d'une unité supposée.

## Phase 1 — Obtenir et figer un modèle exploitable

Cette phase est bloquante : sans poids entraînés, l'architecture seule ne peut
faire aucune prédiction utile.

### Option A — Récupérer le modèle publié

Option prioritaire si les auteurs ont encore rendu le fichier disponible :

1. retrouver le modèle étendu `0_model_extended.h5` associé au papier ;
2. vérifier son origine et ses conditions de réutilisation ;
3. charger le fichier dans l'environnement historique ;
4. inspecter ses formes d'entrée et de sortie ;
5. associer explicitement les 37 sorties à la liste de `smarts.py` ;
6. calculer et enregistrer son empreinte SHA-256 ;
7. produire au moins une prédiction de référence.

### Option B — Réentraîner le modèle

À retenir si le modèle original est introuvable ou incompatible :

1. rétablir l'acquisition des données NIST et SDBS ;
2. corriger le prétraitement historique et tracer les exemples rejetés ;
3. séparer les données **par molécule/InChI**, et non simplement par spectre,
   afin d'éviter les fuites entre entraînement et test ;
4. recréer les labels SMARTS et vérifier manuellement un échantillon ;
5. entraîner d'abord un modèle de référence sans augmentation ;
6. évaluer sur un jeu de molécules totalement indépendant ;
7. publier les poids, leur empreinte et toutes les métadonnées nécessaires.

### Manifeste minimal du modèle

Le fichier `models/manifest.json` devra contenir :

```json
{
  "model_id": "cnn-ir-original-37fg",
  "model_file": "0_model_extended.h5",
  "sha256": "...",
  "framework": "keras",
  "input_shape": [600, 1],
  "x_min_cm-1": 400.0,
  "x_max_cm-1": 4000.0,
  "y_representation": "transmittance_fraction",
  "labels_file": "labels.json",
  "default_threshold": 0.5
}
```

### Critère de fin

Une fonction minimale charge le modèle sur une installation propre, confirme
les formes `(None, 600, 1) → (None, 37)` et reproduit une sortie de référence à
une tolérance numérique documentée.

## Phase 2 — Extraire un prétraitement d'inférence fiable

Le prétraitement utilisateur ne doit pas appeler directement
`scripts/preprocessing.py`, car ce fichier est conçu pour construire les jeux
NIST/SDBS complets.

### Travail

Créer une fonction pure de type :

```python
prepare_ir_spectrum(
    path,
    x_column="wavenumber",
    y_column="intensity",
    x_unit="cm-1",
    y_unit="transmittance_fraction",
) -> PreparedSpectrum
```

Elle devra :

1. lire le fichier sans modifier les données sources ;
2. convertir les colonnes en valeurs numériques ;
3. retirer les lignes non finies en les comptabilisant ;
4. convertir explicitement les unités ;
5. trier l'axe spectral ;
6. fusionner les nombres d'onde dupliqués selon une règle documentée ;
7. contrôler la couverture de la région 400–4000 cm⁻¹ ;
8. traiter les bords selon une politique explicite ;
9. interpoler linéairement sur 600 points décroissants ;
10. borner la transmittance entre 0 et 1 en signalant toute correction ;
11. retourner un tableau `float32` de forme `(1, 600, 1)` ;
12. produire un rapport de contrôle qualité.

### Politique de couverture

L'extrapolation constante utilisée implicitement dans le code historique peut
introduire de longues zones artificielles. Le comportement recommandé est :

- couverture complète 400–4000 cm⁻¹ : accepter ;
- petite portion manquante, sous une tolérance configurable : compléter et
  ajouter un avertissement ;
- portion importante manquante : refuser par défaut ;
- autoriser le forçage uniquement avec une option explicite.

### Contrôles qualité

Le rapport devra au minimum signaler :

- nombre de points initiaux et valides ;
- étendue spectrale réelle ;
- valeurs minimales et maximales avant/après conversion ;
- proportion de valeurs supprimées, bornées ou extrapolées ;
- ordre original de l'axe ;
- présence de doublons ;
- éventuelle incompatibilité d'unité.

### Compatibilité historique

Deux modes pourront être utiles :

- `legacy` : reproduire aussi fidèlement que possible le traitement du modèle
  publié ;
- `strict` : validation plus sûre des données utilisateur.

Toute différence entre ces modes devra être testée et documentée. Il faut
éviter d'« améliorer » arbitrairement le prétraitement si cela éloigne les
données de la distribution utilisée pour entraîner le modèle.

### Critère de fin

Des fichiers équivalents en absorbance, transmittance fractionnaire et
pourcentage de transmittance produisent le même vecteur à la tolérance choisie.

## Phase 3 — Construire l'API d'inférence

### Travail

Implémenter une API Python indépendante de l'interface en ligne de commande :

```python
predict_spectrum(
    spectrum,
    model,
    thresholds=0.5,
) -> PredictionResult
```

Cette API devra :

1. valider la forme et le type du tenseur ;
2. charger le modèle une seule fois pour un lot de fichiers ;
3. vérifier que le modèle possède exactement 37 sorties ;
4. préserver l'ordre canonique des labels ;
5. appliquer soit le seuil uniforme de 0,5, soit 37 seuils nommés ;
6. retourner toutes les probabilités, y compris celles sous le seuil ;
7. enregistrer la version et l'empreinte du modèle dans le résultat ;
8. rendre les résultats déterministes sur CPU dans la mesure du possible.

Le bug historique utilisant `optimal_thresh[0]` pour toutes les classes ne doit
pas être reproduit : le seuil de la classe `i` doit être appliqué à la sortie
`i`.

### Format de sortie

Deux formats seront supportés :

- JSON détaillé pour une application ou une API ;
- CSV aplati pour l'analyse de plusieurs spectres.

Les sorties devront distinguer clairement :

- la probabilité brute ;
- le seuil appliqué ;
- la décision binaire ;
- les avertissements de qualité du spectre.

### Critère de fin

Une prédiction complète peut être obtenue depuis Python sans dépendre des
scripts d'entraînement ni des datasets NIST/SDBS.

## Phase 4 — Fournir une CLI utilisable

### Commandes minimales

```bash
# Inspecter un spectre sans lancer le modèle
cnn-ir inspect mon_spectre.csv --y-unit absorbance

# Prédire un spectre
cnn-ir predict mon_spectre.csv --y-unit absorbance

# Enregistrer le résultat
cnn-ir predict mon_spectre.csv \
  --y-unit absorbance \
  --output predictions.json

# Traiter un dossier
cnn-ir predict-batch data/ \
  --glob "*.csv" \
  --y-unit percent_transmittance \
  --output predictions.csv
```

### Options importantes

- noms des colonnes x et y ;
- unités x et y ;
- chemin ou identifiant du modèle ;
- seuil global ou fichier de seuils par classe ;
- mode `legacy` ou `strict` ;
- sortie JSON ou CSV ;
- niveau de verbosité ;
- option explicite pour accepter une couverture incomplète.

### Comportement attendu

- les erreurs doivent expliquer comment corriger le fichier ;
- les avertissements ne doivent pas être cachés dans une longue trace Python ;
- une erreur sur un fichier d'un lot ne doit pas supprimer les résultats des
  autres fichiers ;
- le code de sortie du processus doit distinguer succès, avertissement et
  échec.

### Critère de fin

Après installation, une personne ne connaissant pas le code obtient une
prédiction et un fichier de résultat avec une seule commande documentée.

## Phase 5 — Tester et valider scientifiquement

### Tests unitaires

Ajouter des tests pour :

- axes croissants et décroissants ;
- valeurs dupliquées et non numériques ;
- conversion absorbance/transmittance ;
- pourcentages de transmittance ;
- spectres partiels ;
- interpolation et forme `(1, 600, 1)` ;
- ordre immuable des 37 labels ;
- seuil global et seuils par classe ;
- fichiers inexistants ou colonnes incorrectes.

### Tests d'intégration

- charger le vrai modèle ;
- prédire un fichier d'exemple ;
- vérifier les 37 probabilités ;
- comparer à une sortie de référence enregistrée ;
- exécuter la CLI sur un lot contenant un fichier valide et un fichier invalide.

### Validation chimique

Constituer un petit benchmark externe comprenant :

- des molécules simples avec groupes clairement connus ;
- des molécules multifonctionnelles ;
- des classes fréquentes et rares ;
- si possible, plusieurs modes de mesure pour une même substance.

Évaluer au minimum précision, rappel, F1, Average Precision et taux de faux
positifs par classe. Ne pas valider le système uniquement sur des spectres ayant
servi à l'entraînement.

### Tests de décalage de domaine

Comparer les performances selon :

- ATR contre transmission/KBr ;
- instrument et résolution ;
- solide, liquide et solution ;
- ligne de base, bruit et saturation ;
- plage spectrale disponible.

Cette étape déterminera si une calibration, un ajustement des seuils ou un
réentraînement sur les données utilisateur est nécessaire.

### Critère de fin

Les limites du système sont mesurées sur des données indépendantes et les
résultats de la CLI sont couverts par des tests automatisés.

## Phase 6 — Moderniser l'installation

Cette phase peut commencer en parallèle des tests, mais ne doit pas modifier les
prédictions de référence sans explication.

### Travail

1. créer un `pyproject.toml` et un package installable ;
2. séparer les dépendances d'inférence des dépendances d'entraînement ;
3. remplacer PlaidML par une version actuelle de TensorFlow/Keras, ou convertir
   le modèle vers un format plus portable ;
4. choisir et documenter les versions de Python supportées ;
5. ajouter un verrou de dépendances ou un environnement reproductible ;
6. tester l'installation sur CPU sans GPU obligatoire ;
7. ajouter une intégration continue pour les tests et le linting.

### Groupes de dépendances proposés

```text
inference : numpy, pandas, scipy, tensorflow/keras
formats   : jcamp (optionnel)
training  : scikit-learn, scikit-optimize, rdkit, iterative-stratification
dev       : pytest, ruff, mypy
```

### Critère de fin

Une installation propre sur un Python supporté permet d'exécuter l'exemple en
CPU, sans installer les dépendances de scraping ou d'entraînement.

## Phase 7 — Documenter l'utilisation et l'interprétation

Mettre à jour le README avec :

- une installation rapide ;
- le téléchargement ou l'emplacement du modèle ;
- un exemple de CSV ;
- une commande d'inférence complète ;
- un exemple de résultat ;
- la signification des probabilités et des seuils ;
- les modes de mesure connus ou non couverts ;
- les limites scientifiques et la procédure de citation.

Ajouter également :

- un exemple exécutable dans `examples/` ;
- une documentation des 37 labels et de leur ordre ;
- une section de dépannage ;
- une politique de versionnement des modèles ;
- une note indiquant qu'une probabilité n'est pas une certitude chimique.

### Critère de fin

La documentation permet d'installer et de tester le projet sans lire les
anciens scripts de recherche.

## Phase 8 — Extensions après le MVP

Ces éléments ne sont pas nécessaires pour la première version utilisable :

- lecture directe de JCAMP-DX ;
- prise en charge de plusieurs conventions de CSV instrumentales ;
- visualisation du spectre original et du spectre interpolé ;
- comparaison réactif/produit par variation des probabilités ;
- petite interface web locale ;
- API REST ;
- estimation d'incertitude et détection hors distribution ;
- recalibration des probabilités ;
- fine-tuning sur des spectres propres au laboratoire ;
- attribution approximative des régions influençant chaque sortie.

Une comparaison réactif/produit devra être présentée comme une aide : la baisse
de la probabilité d'un groupe et l'augmentation d'un autre ne constituent pas à
elles seules une preuve de conversion chimique.

## Ordre de réalisation recommandé

| Priorité | Phase | Résultat |
|---:|---|---|
| 1 | Phase 0 | formats et unités des données connus |
| 2 | Phase 1 | modèle récupéré ou stratégie de réentraînement validée |
| 3 | Phase 2 | prétraitement fiable vers `(1, 600, 1)` |
| 4 | Phase 3 | API Python produisant 37 probabilités |
| 5 | Phase 4 | commande utilisable sur un CSV ou un dossier |
| 6 | Phase 5 | comportement et limites validés |
| 7 | Phase 6 | installation moderne et reproductible |
| 8 | Phase 7 | documentation utilisateur complète |
| 9 | Phase 8 | fonctionnalités avancées |

Les phases 0 à 4 constituent le **MVP d'inférence**. Les phases 5 à 7 sont
nécessaires avant de considérer l'outil comme fiable et partageable.

## Définition de « terminé » pour le MVP

Le premier objectif sera atteint lorsque toutes les affirmations suivantes
seront vraies :

- un modèle entraîné, identifié et vérifié est disponible localement ;
- un CSV utilisateur est validé et transformé en `(1, 600, 1)` ;
- les unités absorbance, transmittance et pourcentage sont gérées explicitement ;
- les 37 probabilités sont associées aux bons labels ;
- le seuil est appliqué correctement à chaque classe ;
- les résultats incluent les avertissements de qualité ;
- une commande documentée traite un fichier et un dossier ;
- des tests couvrent le prétraitement, le chargement et l'inférence ;
- une prédiction de référence est reproductible sur CPU ;
- les limites liées au mode de mesure et au domaine d'entraînement sont visibles
  par l'utilisateur.

## Décisions à prendre avant l'implémentation

Les réponses aux questions suivantes détermineront les premières tâches :

1. Dans quel format exact les spectres utilisateur sont-ils disponibles ?
2. Quelles sont les unités x et y des exports ?
3. Les mesures sont-elles ATR, transmission, KBr ou un mélange ?
4. La plage 400–4000 cm⁻¹ est-elle couverte ?
5. Le modèle publié peut-il être récupéré et redistribué légalement ?
6. Faut-il traiter un fichier à la fois ou des lots complets ?
7. Les sorties doivent-elles être consommées par un humain, un notebook ou une
   autre application ?
8. Des spectres de référence avec groupes fonctionnels connus sont-ils
   disponibles pour la validation ?

La réponse la plus urgente concerne le modèle : si les poids publiés ne sont
pas récupérables, le projet bascule d'un chantier d'inférence relativement
court vers un chantier de collecte, nettoyage et réentraînement beaucoup plus
important.
