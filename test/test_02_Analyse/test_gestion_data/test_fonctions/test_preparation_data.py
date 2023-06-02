##########################
#        IMPORTS         #
##########################
import os
import configparser
from gestion_data.fonctions.preparation_data import step_0_lecture_data_brut

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

chemin_config = os.path.join(repertoire_projet, '02_Analyse/Configuration/config.ini')
config = configparser.ConfigParser()
config.read(chemin_config)

##########################
#       VARIABLES        #
##########################


path_data_brut = f"{repertoire_projet}/{config.get('PATH', 'data_brut')}"

##########################
#       Fonctions        #
##########################


def test_step_0_lecture_csv():
    print("OK")
    print(path_data_brut)
    print("KOKOK")
    var1, var2 = step_0_lecture_data_brut(path_data_brut)
    print(var1)
    print(var2)
