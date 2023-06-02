# Prévenir les Bad Buzz pour les restaurants

Ce projet a pour but d'anticiper les bad buzz à partir de l'analyse de tweet. 

## Décomposition du projet

Ce projet est divisé en 3 grosses parties :
1) Le dossier `01_Preambule` contenant le notebook dans lequel a été développé l'essentiel du projet, à savoir,
la partie traitement des données, développement et entraînement des modèles de prédiction sur les tweet.
2) Le dossier `02_Analyse` qui contient reprend en partie le travail effectué dans le notebook, 
développe un peu plus la gestion des données et les modèles principaux afin d'en amméliorer la modularité et le suivi.
C'est à dire la partie traitement des données, développement et entraînement des modèles de prédiction sur les tweet.
3) Le dossier `03_Production` concernant la partie mise en production du modèle Bert sur streamlit. 

Enfin, le dossier `test` comporte quelques TU élémentaires qui ont
servies au développement de fonctions adaptées.

## Lancement (en local) de la partie 03_Production correspondant au déploiement du modèle

Ouvrez le terminal et lancer la commande suivante :

*   `streamlit run .\03_Production\__main__.py`
