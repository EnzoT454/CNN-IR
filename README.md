# CNN-IR — Identification de groupes fonctionnels à partir de spectres IR

CNN-IR est un projet de recherche qui utilise un réseau de neurones convolutif
unidimensionnel (CNN 1D) pour identifier les groupes fonctionnels présents dans
une molécule à partir de son spectre infrarouge (IR).

Le problème est formulé comme une **classification multilabel** : une molécule
peut contenir plusieurs groupes fonctionnels et le modèle produit donc 37
probabilités indépendantes. Le projet couvre la préparation des spectres,
l'étiquetage chimique, la séparation des données, l'augmentation, la recherche
d'hyperparamètres, l'entraînement et l'évaluation.

Ce dépôt accompagne les travaux décrits dans l'article suivant :

> G. Jung, S. G. Jung et J. M. Cole, *Automatic materials characterization from
> infrared spectra using convolutional neural networks*, Chemical Science,
> 2023, 14, 3600–3609. DOI :
> [10.1039/D2SC05892H](https://doi.org/10.1039/D2SC05892H).

## Principe

Le pipeline transforme un spectre IR en un vecteur binaire indiquant la
présence ou l'absence de chaque groupe fonctionnel :

```text
Spectres NIST et SDBS
        │
        ▼
Conversion en transmittance et interpolation sur 600 points
        │
        ▼
Structure InChI ──SMARTS/RDKit──► 37 labels binaires
        │
        ▼
Séparation entraînement / validation / test
        │
        ▼
CNN 1D
        │
        ▼
37 probabilités ──seuil──► groupes fonctionnels prédits
```

Le modèle ne détermine pas l'identité complète d'une molécule. Il prédit des
éléments de structure tels que « alcool », « arène » ou « ester ».

## Architecture du modèle

Chaque spectre est représenté par 600 intensités couvrant l'intervalle
400–4000 cm⁻¹, puis remis en forme en `(600, 1)` pour le CNN.

```text
Entrée (600 × 1)
  → Conv1D(31 filtres, noyau 11)
  → BatchNorm → ReLU → MaxPool
  → Conv1D(62 filtres, noyau 11)
  → BatchNorm → ReLU → MaxPool
  → Flatten
  → Dense(4927) → Dropout(0,486)
  → Dense(2785) → Dropout(0,486)
  → Dense(1574) → Dropout(0,486)
  → Dense(37, sigmoid)
```

La sortie utilise une activation sigmoïde, et non un softmax, car plusieurs
classes peuvent être vraies simultanément. La fonction de coût principale est
la binary cross-entropy. Une variante pondérée est également fournie pour
étudier le déséquilibre entre classes positives et négatives.

## Groupes fonctionnels

Les labels sont générés automatiquement à partir des structures InChI grâce à
des motifs SMARTS et à RDKit. Les 37 classes sont :

- alcane, alcène, alcyne, arène et halogénoalcane ;
- alcool, aldéhyde, cétone, acide carboxylique, anhydride d'acide, halogénure
  d'acyle, ester et éther ;
- amine, amide, nitrile, imide, imine, composé azoïque, hydrazine, énamine,
  hydrazone et carbamate ;
- thiol, thial, sulfone, acide sulfonique, sulfonamide, sulfonate, sulfoxyde,
  thioamide et sulfure ;
- énol, phénol, isocyanate, isothiocyanate et phosphine.

Les définitions exactes et leur ordre sont disponibles dans
[`scripts/smarts.py`](scripts/smarts.py). Cet ordre doit rester identique entre
le prétraitement, l'entraînement et l'interprétation des prédictions.

## Structure du dépôt

```text
CNN-IR/
├── README.md
├── environment.yml             # environnement principal Keras/PlaidML
├── environment2.yml            # environnement TensorFlow du modèle pondéré
├── requirements.txt            # dépendances de l'environnement principal
├── requirements2.txt           # dépendances du modèle pondéré
├── nist_dataset/
│   └── ids.txt                 # identifiants NIST, sans les spectres
├── sdbs_dataset/
│   └── sdbs_ids.txt            # identifiants SDBS, sans les spectres
└── scripts/
    ├── preprocessing.py         # harmonisation et étiquetage des spectres
    ├── split_data.py            # séparation stratifiée multilabel
    ├── data_augmentation.py     # oversampling et augmentations
    ├── hyperparameter_optimization.py
    ├── train_model.py           # modèles principal, original et augmentés
    ├── train_weighted_model.py  # modèle avec perte pondérée
    ├── optimal_thresholding.py  # seuil de décision par classe
    ├── evaluation.py            # F1, précision, rappel, AP et EMR
    └── smarts.py                # définitions SMARTS des classes
```

Les dossiers de données traitées, modèles et résultats ne sont pas versionnés.
Ils sont créés ou attendus par les scripts sous les noms suivants :

```text
processed_dataset/   augmented_dataset/   models/
searched_parameters/ checkpoints/
```

## Données

L'étude originale combine des spectres issus de :

- [NIST Chemistry WebBook](https://webbook.nist.gov/) ;
- [SDBS — Spectral Database for Organic Compounds](https://sdbs.db.aist.go.jp/).

Le papier indique 50 936 spectres pour 30 611 molécules uniques. Les fichiers
`ids.txt` présents dans ce dépôt sont seulement des listes d'identifiants : ils
ne contiennent ni les spectres, ni les InChI, ni le jeu de données final.

Le prétraitement attend l'arborescence suivante :

```text
nist_dataset/
├── inchi/                       # fichiers <id>.inchi
└── jdx/                         # spectres JCAMP-DX

sdbs_dataset/
├── gif/                         # images de spectres
├── png/                         # images converties par preprocessing.py
└── other/                       # métadonnées contenant les InChI
```

Les scripts de téléchargement `nist_scraper.py` et `sdbs_scraper.py` mentionnés
dans la documentation originale ne sont pas présents dans cette copie du
dépôt. Les données doivent donc être obtenues séparément, dans le respect des
conditions d'utilisation de NIST et SDBS.

## Dépendances

Le projet repose sur une pile logicielle datant de l'étude originale :

- Python 3.7 ;
- NumPy 1.21.6, pandas 1.3.5 et SciPy 1.7.3 ;
- Keras/PlaidML pour le modèle principal ;
- TensorFlow 2.10 pour la variante pondérée ;
- scikit-learn 1.0.2 et scikit-optimize 0.9.0 ;
- RDKit 2022.3.5 ;
- OpenCV, Pillow et `jcamp` pour le traitement des spectres ;
- `iterative-stratification` pour les séparations multilabel.

Les fichiers d'environnement ont été produits sur macOS et contiennent des
dépendances spécifiques à cette plateforme. Conda est la méthode d'installation
recommandée pour reproduire au mieux l'environnement historique.

### Environnement principal

```bash
conda env create --file environment.yml
conda activate fg_predict
```

Alternative avec `pip`, dans un environnement Python 3.7 isolé :

```bash
python -m pip install -r requirements.txt
```

### Environnement du modèle pondéré

```bash
conda env create --file environment2.yml
conda activate fg-plaidml
```

Alternative :

```bash
python -m pip install -r requirements2.txt
```

## Exécution du pipeline

Les chemins des scripts sont relatifs au dossier `scripts`. Les commandes
ci-dessous doivent donc être exécutées depuis ce dossier.

Créez d'abord les dossiers de sortie manquants depuis la racine du dépôt :

```bash
mkdir -p processed_dataset augmented_dataset models searched_parameters checkpoints
cd scripts
```

Puis exécutez les étapes nécessaires :

```bash
# 1. Convertir les données brutes et générer les labels SMARTS
python preprocessing.py

# 2. Créer le test set et les quatre plis entraînement/validation
python split_data.py

# 3. Créer les variantes augmentées
python data_augmentation.py

# 4. Rechercher les hyperparamètres (optionnel et coûteux)
python hyperparameter_optimization.py

# 5. Entraîner les modèles
python train_model.py

# 6. Entraîner la variante pondérée dans le second environnement
conda activate fg-plaidml
python train_weighted_model.py

# 7. Calculer les seuils par classe puis évaluer
conda activate fg_predict
python optimal_thresholding.py
python evaluation.py
```

`train_model.py` n'entraîne pas seulement le modèle principal : si le jeu
augmenté est disponible, il entraîne aussi les modèles original, de contrôle
et d'augmentation pour les proportions 25 %, 50 %, 75 % et 100 %. Son exécution
complète est donc coûteuse en temps et en mémoire.

## Fichiers produits

| Étape | Sortie principale |
|---|---|
| Prétraitement | `processed_dataset/input_dataset.csv` et `label_dataset.csv` |
| Séparation | `processed_dataset/processed_dataset.pickle` |
| Augmentation | `augmented_dataset/augmented_dataset.pickle` |
| Optimisation | paramètres dans `searched_parameters/`, modèles dans `checkpoints/` |
| Entraînement | modèles Keras HDF5 (`*.h5`) |
| Évaluation | tableaux de métriques affichés dans le terminal |

Chaque entrée du CNN contient 600 intensités. Les CSV intermédiaires ajoutent
aussi l'InChI et l'origine du spectre comme métadonnées ; ces colonnes sont
retirées avant l'entraînement.

## Métriques

L'évaluation calcule notamment :

- le F1, la précision et le rappel de chaque groupe fonctionnel ;
- l'Average Precision (AP), adaptée aux classes positives rares ;
- l'Exact Match Rate (EMR), qui exige que toutes les classes d'une molécule
  soient simultanément correctes ;
- les accuracies séparées pour la présence et l'absence des groupes.

Dans l'article, le modèle étendu atteint un F1 macro moyen de 0,85, un F1
pondéré de 0,93, une mean Average Precision de 0,88 et un EMR global voisin de
0,72. Ces valeurs sont les résultats publiés, pas des résultats reproduits à
partir de cette copie du dépôt.

## État actuel et limitations connues

Le dépôt contient le code expérimental, mais **le pipeline complet n'est pas
exécutable tel quel** sans récupérer les données, les modèles et les scripts de
collecte manquants. Les points suivants devront également être corrigés pour
obtenir une reproduction fiable :

- `evaluation.py` contient un chemin de modèle invalide (`..Calculating/...`) ;
- l'application des seuils optimaux utilise le seuil de la première classe pour
  toutes les classes au lieu d'utiliser un seuil par classe ;
- `vertical_aug()` référence une variable `constant1` non définie ;
- une branche de `convert_x()` référence `x_out` avant son initialisation ;
- plusieurs exceptions génériques masquent les erreurs de prétraitement ;
- les emplacements de sauvegarde et de chargement des modèles ne sont pas
  cohérents entre tous les scripts ;
- la séparation est effectuée au niveau des spectres et ne garantit pas que les
  différents spectres d'une même molécule restent dans le même ensemble ;
- aucune suite de tests automatisés n'est fournie ;
- la pile Python/Keras/PlaidML est ancienne et peut être difficile à installer
  sur un système moderne.

La prochaine étape recommandée est de corriger ces problèmes, de centraliser
les chemins dans une configuration unique et d'ajouter une commande d'inférence
acceptant un nouveau spectre.

## Licence et citation

Aucun fichier de licence n'est actuellement inclus dans ce dépôt. Avant de
redistribuer ou de réutiliser le code, vérifiez les conditions du dépôt source
et celles des bases NIST et SDBS.

Pour citer la méthode scientifique, utilisez l'article associé :

```bibtex
@article{jung2023automatic,
  title   = {Automatic materials characterization from infrared spectra using convolutional neural networks},
  author  = {Jung, Guwon and Jung, Son Gyo and Cole, Jacqueline M.},
  journal = {Chemical Science},
  year    = {2023},
  volume  = {14},
  pages   = {3600--3609},
  doi     = {10.1039/D2SC05892H}
}
```
