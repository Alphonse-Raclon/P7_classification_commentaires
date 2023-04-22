# Fichier servant à effectuer des tests

#######################
#       Imports       #
#######################
import os
import pandas as pd
from Production.python.pretraitement import normalisation_texte

#######################
#      Variables      #
#######################

source = os.getcwd()
comm = "{}/Production/ressources/commentaire.csv".format(source)

data_brut = pd.read_csv(comm, sep=",", header=None,
                        names=["target", 'ids', 'date', 'flag', 'user', 'text'],
                        on_bad_lines='skip', encoding_errors='ignore',
                        encoding="ISO-8859-1")

data = data_brut[["target", "text"]]


#######################
#      Fonctions      #
#######################

def test_normalisation_texte():
    data_clean = normalisation_texte(data)

    print(data_clean.sample(5))
    print(data_clean.columns)
