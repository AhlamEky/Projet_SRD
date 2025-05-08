Projet de Classification d'Images

Description du Projet:

Ce projet consiste à construire un modèle de classification d'images de véhicules.
Nous utilisons un réseau de neurones convolutifs (CNN) pour prédire le type de véhicule
à partir d'une image. Le projet inclut des étapes de prétraitement des données,
d'entraînement du modèle, et de déploiement via une interface web.

À propos du Dataset

Contexte:
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


Préparation des Données

Téléchargement du Dataset:

Le dataset est téléchargé et placé dans le dossier data/raw/.

Prétraitement des Données

Les images brutes sont traitées dans le script data_preprocessing.py.
Ce script effectue les actions suivantes :

Redimensionnement des images à une taille uniforme de 224x224 pixels.

Augmentation des données avec des rotations et des retournements horizontaux pour améliorer la robustesse du modèle.

Normalisation des images pour les adapter à l'input du modèle (valeurs entre -1 et 1).

Division du dataset en trois sous-ensembles : entraînement (train), validation (val), et test (test).

Pour exécuter le prétraitement, on lance la commande suivante :

'python scripts/data_preprocessing.py'
Cela générera les données prétraitées dans le dossier data/processed/.

Chargement des Données
Le script dataset_loader.py crée des DataLoaders pour charger les images
dans le modèle pendant l'entraînement.
Les données sont séparées en trois ensembles : train, val, et test.
