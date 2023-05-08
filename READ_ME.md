# Prévenir les Bad Buzz pour les restaurants

Ce projet a pour but d'anticiper les bad buzz à partir de l'analyse de tweet. 

## Décomposition du projet

Ce projet est divisé en 2 grosses parties :
1) Le dossier Analyse qui contient le jupyter notebook `P7_01_commentaires_prediction_buzz.ipynb` dans lequel est effectué 
toute la partie analyse.
C'est à dire la partie traitement des données, développement et entraînement des modèles de prédiction sur les tweet.
2) Le dossier production concernant la partie MLFlow. Ce dossier contient toute la partie mise en production pour 
exploter le meilleur modèle développé dans la partie analyse. Le dossier test comporte quelques TU élémentaires qui ont
servies au développement de fonctions adaptées à la partie MLFlow.

## Lancement du projet partie MLFLOW

Ouvrez le terminal et lancer la commande suivante :

*   `streamlit run .\Production\__main__.py`
