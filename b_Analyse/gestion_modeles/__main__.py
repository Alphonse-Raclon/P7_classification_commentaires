# Scrip principal lançant toute la partie entraînement des modèles

##########################
#        IMPORTS         #
##########################
import itertools
from b_Analyse.gestion_modeles.fonctions.models import *

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

chemin_config = os.path.join(repertoire_projet, 'b_Analyse/Configuration/config.ini')
config = configparser.ConfigParser()
config.read(chemin_config)

##########################
#      Variables         #
##########################

file_entrainement = os.path.join(repertoire_projet, config.get('PATH_MODELS', 'file_entrainement'))
file_evaluation = os.path.join(repertoire_projet, config.get('PATH_MODELS', 'file_evaluation'))


##########################
#       Fonction         #
##########################
def main(file):
    """
    Cette fonction lance l'entraînement des différents modèles en faisant varier les paramètres

    :param file: path des datas, fichier csv avec les données d'entrainement
    """

    ################################
    # Partie regression logistique #
    param_reg_log = {'embedding': [0, 1],
                     'sg': [0, 1],
                     'vect_size': [100, 150],
                     'min_count': [1, 3, 5]}

    # Obtention de toutes les combinaisons possibles de paramètres
    param_combinations = list(itertools.product(*param_reg_log.values()))

    # Appel de la fonction model_regression_logistique pour chaque combinaison de paramètres
    for params in param_combinations:
        model_regression_logistique(file, **dict(zip(param_reg_log.keys(), params)))

    ##########################
    # Partie foret aleatoire #

    param_foret_aleatoire = {
        'embedding': [0, 1],
        'sg': [0, 1],
        'vect_size': [100],
        'min_count': [1, 3, 5],
        'n_estimators': [100],
        'max_depth': [5, 10],
        'max_samples': [0.6, 0.8, 1.0]
    }

    param_combinations = list(itertools.product(*param_foret_aleatoire.values()))

    for params in param_combinations:
        model_foret_aleatoire(file, **dict(zip(param_foret_aleatoire.keys(), params)))

    #################################
    # Partie gradient boosting tree #

    param_gradient_boosting = {
        'embedding': [0, 1],
        'sg': [0, 1],
        'vect_size': [100, 150],
        'min_count': [1, 3, 5],
        'n_estimators': [60],
        'max_depth': [5, 10],
        'subsample': [0.6, 0.8, 1.0]
    }

    param_combinations = list(itertools.product(*param_gradient_boosting.values()))

    for params in param_combinations:
        model_gradient_boosting_tree(file, **dict(zip(param_gradient_boosting.keys(), params)))


def main_bert(file):
    """ Cette fonction juste pour avoir les performances d'un modèle Bert déjà entraîné"""
    model_bert(file)


if __name__ == '__main__':
    main(file_entrainement)
    # main_bert(file_evaluation)
