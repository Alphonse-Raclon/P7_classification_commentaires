# script lançant la partie récuperation des datas


##########################
#        IMPORTS         #
##########################
import pandas as pd
import os
import configparser

##########################
#     CONFIGURATION      #
##########################
repertoire_courant = os.path.dirname(os.path.abspath(__file__))
# Remonter les répertoires parents jusqu'à atteindre le répertoire racine du projet
while not os.path.basename(repertoire_courant) == 'P7_classification_commentaires':
    repertoire_parent = os.path.dirname(repertoire_courant)
    if repertoire_parent == repertoire_courant:
        raise FileNotFoundError("Répertoire racine du projet introuvable.")
    repertoire_courant = repertoire_parent
chemin_config = os.path.join(repertoire_courant, '02_Analyse/Configuration/config.ini')
config = configparser.ConfigParser()
config.read(chemin_config)

##########################
#       VARIABLES        #
##########################

download_url = config.get('MEGA', 'mega_link')

##########################
#       Fonctions        #
##########################

