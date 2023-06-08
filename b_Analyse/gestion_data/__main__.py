# script lançant la partie récuperation des datas


##########################
#        IMPORTS         #
##########################
import configparser

from b_Analyse.gestion_data.fonctions.preparation_data import *


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
chemin_config = os.path.join(repertoire_courant, 'b_Analyse/Configuration/config.ini')
config = configparser.ConfigParser()
config.read(chemin_config)

##########################
#       VARIABLES        #
##########################
path_data_brut = os.path.join(repertoire_courant, config.get('PATH_GESTION_DATA', 'path_data_brut'))
path_data_transformee = os.path.join(repertoire_courant, config.get('PATH_GESTION_DATA', 'path_data_transformee'))

##########################
#       Fonctions        #
##########################


def main(path_data_input, path_data_output):
    """
    Cette fonction main récupère les données brut et les prépare pour
    être prête à être utilisée pour les modèles de machine learning

    :param path_data_input: path menant aux données brutes
    :param path_data_output: path où les données normalisées seront enregistrées

    La fonction enregistre les données prêtes à l'emploi par paquets
    """

    # Recuperation des données d'intérêt uniquement
    df_0, df_4 = step_0_lecture_data_brut(path_data_input)

    # Normalisation des données
    df_0["text"], df_4["text"] = step_1_3_main_normalisation_texte(df_0["text"]), \
        step_1_3_main_normalisation_texte(df_4["text"])

    # Enregistrement par lots des données
    step_2_ecriture_lots(df_0, df_4, path_data_output)


if __name__ == '__main__':
    main(path_data_brut, path_data_transformee)
