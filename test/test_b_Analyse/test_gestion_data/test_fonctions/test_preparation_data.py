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


path_data_brut = f"{repertoire_projet}/{config.get('PATH', 'path_data_brut')}"
path_data_transforme_init = "{}/{}".format(repertoire_projet, config.get('PATH', 'path_data_transforme_init'))
path_data_transforme_norm = "{}/{}".format(repertoire_projet, config.get('PATH', 'path_data_transforme_norm'))

TEST_TEMP_DIR = f"{repertoire_projet}/test/test_b_Analyse/test_gestion_data/zone_transformee"


##########################
#       Fonctions        #
##########################

# Fixture pour créer le répertoire temporaire avant chaque test
@pytest.fixture(autouse=True)
def create_test_temp_dir():
    # Créer le répertoire temporaire
    os.makedirs(TEST_TEMP_DIR)

    # Exécuter les tests
    yield

    # Supprimer le répertoire temporaire après chaque test
    shutil.rmtree(TEST_TEMP_DIR)


def test_step_0_lecture_csv():
    """
    Cette fonction test le bon fonctionnement de la fonction eponyme
    """
    path_df_0_expected = "{}/test/ressources/b_Analyse/gestion_data/ressources/" \
                         "zone_brut/df_0_step_0_expected.csv".format(repertoire_projet)

    path_df_4_expected = "{}/test/ressources/b_Analyse/gestion_data/ressources/" \
                         "zone_brut/df_4_step_0_expected.csv".format(repertoire_projet)

    df_0, df_4 = step_0_lecture_data_brut(path_data_brut)

    df_0 = df_0.sort_values(axis=0, by="text").reset_index(drop=True)
    df_4 = df_4.sort_values(axis=0, by="text").reset_index(drop=True)

    df_0_expected = pd.read_csv(path_df_0_expected, names=["target", "text"]).sort_values(axis=0,
                                                                                          by="text").reset_index(
        drop=True)
    df_4_expected = pd.read_csv(path_df_4_expected, names=["target", "text"]).sort_values(axis=0,
                                                                                          by="text").reset_index(
        drop=True)

    assert (df_0_expected == df_0).all().all()
    assert (df_4_expected == df_4).all().all()


def test_step_1_ecriture_lots():
    """
    Cette fonction test le bon fonctionnement de la fonction eponyme
    """
    df_label_0, df_label_4 = step_0_lecture_data_brut(path_data_brut)
    step_2_ecriture_lots(df_label_0, df_label_4, path_data_transforme_init, lot_size=3)

    file_list = os.listdir(path_data_transforme_init)
    file_0 = file_list[0]

    creation_time_1 = os.path.getmtime(os.path.join(path_data_transforme_init, file_0))
    time.sleep(1)

    step_2_ecriture_lots(df_label_0, df_label_4, path_data_transforme_init, lot_size=4)

    creation_time_2 = os.path.getmtime(os.path.join(path_data_transforme_init, file_0))

    assert len(file_list) == 4
    assert creation_time_2 > creation_time_1


def test_step_2_1_clean_text():
    """
    Cette fonction test le bon fonctionnement de la fonction eponyme
    """

    comment = "@johnnybeane hey! You just changed your default. "
    comment_clean = step_1_1_clean_text(comment)

    comment_expected = "AT_USER hey  you just changed your default "

    assert comment_clean == comment_expected
