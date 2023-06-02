# Fichier servant à effectuer des tests

#######################
#       Imports       #
#######################
import os
import pandas as pd
from Production.python.fonction_traitement import normalisation_texte, model_bert

#######################
#      Variables      #
#######################

source = os.getcwd()
comm = "{}/test/ressources/commentaire.csv".format(source)
comm_label = "{}/test/ressources/commentaire_labellise.csv".format(source)

data_label_brut = pd.read_csv(comm_label, sep=",", header=None,
                              names=["target", 'ids', 'date', 'flag', 'user', 'text'],
                              on_bad_lines='skip', encoding_errors='ignore',
                              encoding="ISO-8859-1")

data_label = data_label_brut[["target", "text"]]

data = pd.read_csv(comm,
                   squeeze=True,
                   names=["text"],
                   on_bad_lines='skip', encoding_errors='ignore',
                   encoding="ISO-8859-1")


#######################
#      Fonctions      #
#######################

def test_normalisation_texte():
    data_clean = normalisation_texte(data.sample(5))

    print(data_clean)


def test_model_bert():
    print(model_bert())



