# Ce script contient toutes les fonctions liées à la récupération et au prétraitement des données

##########################
#        IMPORTS         #
##########################
import pandas as pd


##########################
#       Fonctions        #
##########################


def step_0_lecture_data_brut(path_data):
    """
    Ce script récupère un fichier csv ZIPPÉ et récupère les colonnes d'intérêt et le stocke en plusieurs lots
    équilibrés selon les labels

    :param path_data : l'endroit où récupérer le fichier csv
    """
    # Téléchargement du fichier depuis Mega
    response = requests.get(path_data)

    # Chargement fichier csv
    if response.status_code == 200:
        content = response.content

        # Création d'une mémoire tampon à partir du contenu du fichier
        buffer = io.StringIO(content.decode('utf-8'))

        i = 0
        chunksize = 100000

        # Lecture du fichier CSV en chunks avec pandas
        liste_data = []
        for chunk in pd.read_csv(buffer, sep=',', names=["target", "ids", "date", "flag", "user", "text"],
                                 chunksize=chunksize, on_bad_lines='skip', nrows=1600000,
                                 encoding_errors='ignore', encoding="ISO-8859-1"):
            locals()['data_bad_buzz_' + str(i)] = chunk
        #     liste_data.append(chunk)
        #     i += 1
        df = pd.read_csv(buffer, sep="\t")
        print("Chargement des données terminé.")
        return df
    else:
        print("Échec du téléchargement du fichier depuis Mega.")


import pandas as pd


import pandas as pd

def step_0_lecture_data_brut(path_data, chunksize=100000):
    """
    Cette fonction lit un fichier CSV en utilisant des chunks, récupère les colonnes d'intérêt et renvoie
    deux DataFrames distincts pour les labels 0 et 4.

    :param path_data : le chemin vers le fichier CSV
    :param chunksize : la taille des chunks pour la lecture du fichier (par défaut : 100000)
    :return : un tuple contenant les DataFrames pour les labels 0 et 4 respectivement
    """

    # Définition des colonnes d'intérêt
    columns_of_interest = ['target', 'text']

    # Initialisation des DataFrames pour les labels 0 et 4
    df_label_0 = pd.DataFrame(columns=columns_of_interest)
    df_label_4 = pd.DataFrame(columns=columns_of_interest)

    # Lecture du fichier CSV en utilisant des chunks
    for chunk in pd.read_csv(path_data, sep=',', names=["target", "ids", "date", "flag", "user", "text"],
                             chunksize=chunksize, encoding_errors='ignore', encoding="ISO-8859-1"):
        # Filtrage des colonnes d'intérêt (labels et texte)
        df_filtered = chunk[columns_of_interest]

        # Séparation des DataFrames en fonction des labels
        df_label_0_chunk = df_filtered[df_filtered['target'] == 0]
        df_label_4_chunk = df_filtered[df_filtered['target'] == 4]

        # Ajout des données des chunks aux DataFrames principaux
        df_label_0 = pd.concat([df_label_0, df_label_0_chunk])
        df_label_4 = pd.concat([df_label_4, df_label_4_chunk])

    return df_label_0.head(), df_label_4.head()

