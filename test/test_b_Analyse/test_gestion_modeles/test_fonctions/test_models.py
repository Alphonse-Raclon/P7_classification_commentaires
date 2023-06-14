##########################
#        IMPORTS         #
##########################
import os
import time
import pandas as pd
import pytest
import configparser
import shutil
from b_Analyse.gestion_data.fonctions.preparation_data import step_0_lecture_data_brut, step_2_ecriture_lots, \
    step_1_1_clean_text

##########################
#     CONFIGURATION      #
##########################
repertoire_projet = os.path.dirname(os.path.abspath(__file__))
# Remonter les répertoires parents jusqu'à atteindre le répertoire racine du projet
while not os.path.basename(repertoire_projet) == 'P7_classification_commentaires':
    repertoire_parent = os.path.dirname(repertoire_projet)
    if repertoire_parent == repertoire_projet:
        raise FileNotFoundError("Répertoire racine du projet introuvable.")
    repertoire_projet = repertoire_parent

chemin_config = os.path.join(repertoire_projet, 'test/test_b_Analyse/Configuration/config.ini')
config = configparser.ConfigParser()
config.read(chemin_config)

##########################
#       VARIABLES        #
##########################
