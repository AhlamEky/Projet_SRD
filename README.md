#***Projet de Classification d'Images*** (niveau 1)

##- Description du Projet: (niveau 2)

Ce projet consiste à construire un modèle de classification d'images de véhicules.
Nous utilisons un réseau de neurones convolutifs (CNN) pour prédire le type de véhicule
à partir d'une image. Le projet inclut des étapes de prétraitement des données,
d'entraînement du modèle, et de déploiement via une interface web.

- À propos du Dataset

- Contexte:
Vehicle Dataset

Ce dataset est une collection bien structurée d’images de véhicules, 
collectées via le moteur de recherche DuckDuckGo.
Il comprend 20 classes distinctes, correspondant à différents types de
véhicules. Voici la liste des classes disponibles dans le dataset :

Car
Motorcycle
Bicycle
Truck
Bus
Van
Rickshaw
Scooter
Skateboard
Ambulance
Fire Truck
Tractor
Segway
Unicycle
Jet Ski
Helicopter
Airplane
Boat
Kayak


##***Préparation des Données*** (niveau 2)

- Téléchargement du Dataset:
Le dataset est téléchargé et placé dans le dossier data/raw/.
le lien du dataset sur internet est le suivant : https://www.kaggle.com/datasets/mmohaiminulislam/vehicles-image-dataset

- Prétraitement des Données
Les images brutes sont traitées dans le script data_preprocessing.py.
Ce script effectue les actions suivantes :

Redimensionnement des images à une taille uniforme de 224x224 pixels.
Augmentation des données avec des rotations et des retournements horizontaux pour améliorer la robustesse du modèle.
Normalisation des images pour les adapter à l'input du modèle (valeurs entre -1 et 1).
Division du dataset en trois sous-ensembles : entraînement (train), validation (val), et test (test).

Pour exécuter le prétraitement, on lance la commande suivante :
**'python scripts/data_preprocessing.py'**
Cela générera les données prétraitées dans le dossier data/processed/.

- Chargement des Données
Le script dataset_loader.py crée des DataLoaders pour charger les images
dans le modèle pendant l'entraînement.
Les données sont séparées en trois ensembles : train, val, et test.

##***Entraînement Distribué sur Cluster GPU*** (niveau 2)

Pour accélérer l’entraînement et gérer la volumétrie importante des données, nous avons implémenté un entraînement distribué sur un cluster GPU. Cette approche permet de paralléliser le calcul sur plusieurs GPU afin de réduire considérablement le temps d’entraînement.

- Implémentation
Le script distributed_train.py lance plusieurs processus, chacun s’exécutant sur un GPU distinct, identifiés par un rank unique.
La communication entre les GPU est gérée via le backend nccl de PyTorch, optimisé pour les échanges à haute vitesse.
Les données sont réparties entre les différents GPU via un DistributedSampler pour éviter les doublons dans le traitement des batches.
Le modèle CNN est synchronisé entre tous les processus, assurant une cohérence des poids pendant la rétropropagation.
Le modèle entraîné est sauvegardé périodiquement dans le dossier models/.

- Commande de lancement
Pour lancer l’entraînement distribué sur 2 GPU, on exécute simultanément :
**python distributed_train.py --rank 0 --world_size 4
python distributed_train.py --rank 1 --world_size 4
python distributed_train.py --rank 2 --world_size 4
python distributed_train.py --rank 3 --world_size 4**

##***Interface Web*** (niveau 2)

Pour faciliter l’utilisation du modèle par des utilisateurs non techniques, une interface web a été développée.

- Description
L’interface est codée dans app.py
Elle permet à l’utilisateur de téléverser une image de véhicule et d’obtenir en temps réel la prédiction du type de véhicule affichée à l’écran.
L’interface propose aussi une visualisation simple des résultats

- Lancement de l’interface
Pour lancer l’application web, il suffit d’exécuter :
**streamlit run app.py**

- Installation et dépendances
Pour installer les bibliothèques nécessaires, utilisez la commande suivante :
**pip install -r requirements.txt**
