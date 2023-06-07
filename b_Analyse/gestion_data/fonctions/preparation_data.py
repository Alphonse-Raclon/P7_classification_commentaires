# Ce script contient toutes les fonctions liées à la récupération et au prétraitement des données

##########################
#        IMPORTS         #
##########################
import pandas as pd
import os
import glob


##########################
#       Fonctions        #
##########################


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

    return df_label_0.sample(frac=1).reset_index(drop=True), df_label_4.sample(frac=1).reset_index(drop=True)


def step_1_ecriture_lots(df_label_0, df_label_4, output_dir, lot_size=40000):
    """
    Cette fonction répartit les données des DataFrames pour les labels 0 et 4 en lots équilibrés de taille spécifiée
    et les enregistre au format CSV dans un répertoire donné. Le répertoire de sortie est vidé entièrement avant
    l'enregistrement des lots.

    :param df_label_0 : DataFrame contenant les données pour le label 0
    :param df_label_4 : DataFrame contenant les données pour le label 4
    :param output_dir : répertoire de sortie pour les fichiers CSV
    :param lot_size : taille des lots équilibrés
    """

    # Vider le répertoire de sortie
    if os.path.exists(output_dir):
        file_list = os.listdir(output_dir)
        for file_name in file_list:
            file_path = os.path.join(output_dir, file_name)
            os.remove(file_path)
    else:
        os.makedirs(output_dir)

    # Recupération du nombre de lots à faire en fonction du nombre de lignes tout label confondu
    num_lots = (df_label_0.shape[0] + df_label_4.shape[0]) // lot_size

    lot_size_0, lot_size_4 = (df_label_0.shape[0] // num_lots, df_label_4.shape[0] // num_lots)

    # Répartition des données en lots équilibrés
    for i in range(num_lots):
        start_idx_0, start_idx_4 = (i * lot_size_0, i * lot_size_4)
        end_idx_0, end_idx_4 = (start_idx_0 + lot_size_0, start_idx_4 + lot_size_4)
        lot_data_0 = df_label_0.iloc[start_idx_0:end_idx_0]
        lot_data_4 = df_label_4.iloc[start_idx_4:end_idx_4]

        # Concatenation des lots au label différent
        lot_data = pd.concat([lot_data_0, lot_data_4])

        # Enregistrement au format csv des différents lots
        lot_path = os.path.join(output_dir, f"lot_{i}.csv")
        lot_data.to_csv(lot_path, index=False)



