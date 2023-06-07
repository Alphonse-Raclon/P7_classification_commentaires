# Fichier servant à effectuer des tests

#######################
#       Imports       #
#######################
import os
import pandas as pd
# from __main__ import main

#######################
#      Variables      #
#######################
source = os.getcwd()
comm = "{}/test/ressources/bad_buzz_ready4analysis.csv".format(source)

bad_buzz = pd.read_csv(comm, sep="\t")


#######################
#      Fonctions      #
#######################

# def test_dev():
#     print("ok")
#     print(main(bad_buzz))


