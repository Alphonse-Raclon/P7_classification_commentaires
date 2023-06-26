# Prévenir les Bad Buzz pour les restaurants

Ce projet a pour but d'anticiper les bad buzz à partir de l'analyse de tweet. 

## Décomposition du projet

Ce projet est divisé en 3 grosses parties :
1) Le dossier `a_Preambule` contenant le notebook dans lequel a été développé l'essentiel du projet, à savoir,
la partie traitement des données, développement et entraînement des modèles de prédiction sur les tweet.
2) Le dossier `b_Analyse` qui contient reprend en partie le travail effectué dans le notebook, 
développe un peu plus la gestion des données et les modèles principaux afin d'en amméliorer la modularité et le suivi.
C'est à dire la partie traitement des données, développement et entraînement des modèles de prédiction sur les tweet.
3) Le dossier `c_Production` concernant la partie mise en production du modèle Bert sur streamlit. 

Enfin, le dossier `test` comporte quelques TU élémentaires qui ont
servies au développement de fonctions adaptées.


## Partie b_Analyse

Cette partie regroupe 2 sous-parties :

*   gestion_datas : qui se lance avec le `__main__.py` permet de lancer le prétraitement avec le renvoie en sortie 
de données (modulable) prête à l'utilisation pour les modèles de prédiction
*  gestion_modeles : qui se lance avec le `__main__.py` traite de la gestion entraînement et enregistrement des modèles 
dans le mlflow
## Lancement de la partie c_Production correspondant au déploiement du modèle

Pour lancer en local :

Ouvrez le terminal et lancer la commande suivante :

*   `streamlit run .\c_Production\__main__.py`

Ou déployez en accès libre aux adresses suivantes version 2 et 3 :

* https://h-linke-p7-classification-commentair-production--main---11s0jq.streamlit.app/
* https://h-linke-p7-classification-commenta-c-production--main---n8fgp1.streamlit.app/

## Lancement des tests unitaires (en local)

*   `pytest --cov-report html:coverage_report --cov=b_Analyse/gestion_modeles/fonctions/models`
