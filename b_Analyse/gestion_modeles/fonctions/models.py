# Ce script contient toutes les fonctions liées à la mise en place des modèles

##########################
#        IMPORTS         #
##########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import configparser
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import mlflow

from gensim.models import Word2Vec
from gensim.models.fasttext import FastText

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

# Get the current date and time
now = datetime.now()

# Format the date and time as AAAAMMJJhhmmss
formatted_date = now.strftime("%Y%m%d%H%M%S")

path_image = os.path.join(repertoire_projet, config.get('PATH_MODELS', 'path_image'))


##########################
#       Fonctions        #
##########################

# Chargement du fichier de data

def step_util_load_data(file_path):
    """
    Cette fonction charge un fichier CSV dans une série pandas
    :param file_path: fichier CSV
    :return: Série pandas
    """
    return pd.read_csv(file_path)


# Méthodes d'embedding

def step_0_1_word2vec(bad_buzz, sg, vect_size, min_count):
    """
    Méthode : Word2Vec
    Cette fonction prend un DataFrame en entrée contenant
    les commentaires à classifier et procède à l'embedding des mots. Pour obtenir l'embedding du commentaire,
    on utilise la moyenne.

    :param bad_buzz: DataFrame pandas composé des colonnes label et text
    :param sg: paramètre de Word2Vec (0 ou 1)
    :param vect_size: paramètre de Word2Vec
    :param min_count: paramètre de Word2Vec spécifiant le nombre minimum d'occurrences d'un mot pour être pris en compte
    :return: DataFrame correspondant à la matrice composée des vecteurs correspondant à chaque commentaire
    """
    bad_buzz.text = bad_buzz.text.apply(lambda x: x.split(" "))

    # Compter les occurrences de chaque mot
    vocab = Counter()
    sentences = bad_buzz.text.tolist()
    for words in sentences:
        vocab.update(words)
    # Filtrer les mots selon le min_count
    bad_buzz.text = bad_buzz.text.apply(lambda words: [word for word in words if vocab[word] >= min_count])
    bad_buzz.text = bad_buzz.text.apply(lambda words: words if len(words) > 0 else ['<empty_comment>'])

    sentences = bad_buzz.text.tolist()

    # Entraîner le modèle Word2Vec
    model = Word2Vec(sentences, sg=sg, vector_size=vect_size, min_count=1, workers=8)

    # Enregistrement des vecteurs de chaque commentaire
    bad_buzz_word2vec = bad_buzz.text.apply(lambda words: np.mean([model.wv[word] for word in words], axis=0))

    bad_buzz_word2vec = bad_buzz_word2vec.apply(pd.Series)

    return bad_buzz_word2vec


def step_0_2_fasttext(bad_buzz, sg, vect_size, min_count):
    """
    Méthode : FastText
    Cette fonction prend un dataFrame en entrée contenant
    les commentaires à classifier et procède à l'embedding des mots. Pour obtenir l'embedding du commentaire
    on utilise la moyenne.

    :param bad_buzz: DataFrame pandas composé des colonnes label et text
    :param sg: paramètre de Word2Vec (0 ou 1)
    :param vect_size: paramètre de Word2Vec
    :param min_count: paramètre de Word2Vec spécifiant le nombre minimum d'occurrences d'un mot pour être pris en compte
    :return: dataFrame correspondant à la matrice composée des vecteurs correspondant à chaque commentaire
    """
    bad_buzz.text = bad_buzz.text.apply(lambda x: x.split(" "))

    # Entraînement d'un modèle FastText
    model = FastText(sentences=bad_buzz['text'], sg=sg, vector_size=vect_size, window=5, min_count=min_count)

    # Enregistrement des vecteurs de chaque commentaire
    bad_buzz_fasttext = bad_buzz['text'].apply(lambda words: model.wv[words].mean(axis=0))

    # Préparation des données pour former un dataframe
    bad_buzz_fasttext = bad_buzz_fasttext.apply(pd.Series)

    return bad_buzz_fasttext


# Réduction de dimension et séparation des données pour entraînement et test
def step_1_pca_train_test_split(label, matrice_embedding):
    """
    Cette fonction réduit la dimmension de nos vecteurs commentaires à l'aide du pca et
    renvoie des jeux de test et d'entraînement

    :param label: Serie pandas contenant les labels des commentaires
    :param matrice_embedding: DataFrame pandas correspondant aux commentaires sous forme de vecteur
    :return: 4 Series pour le train and test des modèles, X_train, X_test, y_train, y_test
    """
    # Standardisation des données
    scaler = StandardScaler()
    vect_comm_std = scaler.fit_transform(matrice_embedding)

    # Calcul des composantes principales
    n_comp = 0.99

    pca = PCA(n_components=n_comp, random_state=1)
    res_vect_comm = pca.fit_transform(vect_comm_std)

    X = res_vect_comm
    y = label

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test


# Modèles

def step_2_1_regression_logistique(X_train, X_test, y_train):
    """
    Modèle de régression logistique

    :return: Tuple de Liste, liste des prédictions associées aux commentaires et la liste des probas associées aux prédictions
    """
    # Entraîner le modèle
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Faire des prédictions sur les données de test
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # enregistrement mlflow du model
    mlflow.sklearn.log_model(clf, "regression_logistique")

    return y_pred, y_proba


def step_2_2_foret_aleatoire(X_train, X_test, y_train, n_estimators, max_depth, max_samples):
    """
    Modèle de foret aleatoire

    :return: Tuple de Liste, liste des prédictions associées aux commentaires et la liste des probas associées aux prédictions
    """
    # Initialisation et entraînement du modèle de forêt aléatoire
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features="sqrt",
                                max_samples=max_samples, random_state=42, n_jobs=8)
    rf.fit(X_train, y_train)

    # Faire des prédictions sur les données de test
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    # enregistrement mlflow du model
    mlflow.sklearn.log_model(rf, "foret_aleatoire")

    return y_pred, y_proba


def step_2_3_gradient_boosting_tree(X_train, X_test, y_train, n_estimators, max_depth, subsample):
    """
    Modèle de gradient boosting tree

    :return: Tuple de Liste, liste des prédictions associées aux commentaires et la liste des probas associées aux prédictions
    """
    # Initialisation et entraînement du modèle de forêt aléatoire
    gb = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features="sqrt",
                                    subsample=subsample, random_state=42, n_iter_no_change=5)
    gb.fit(X_train, y_train)

    # Faire des prédictions sur les données de test
    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)

    # enregistrement mlflow du model
    mlflow.sklearn.log_model(gb, "gradient_boosting_tree")

    return y_pred, y_proba


def step_2_4_chargement_model_bert():
    """
    Cette fonction charge le modèle de classication bert préalablement entraîné

    :return: modèle bert
    """

    weights_path = "{}/c_Production/model_bert/poids_bert_class_output_v2.npy".format(repertoire_projet)
    # Créer le modèle BERT
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                                     name="bert_preprocess")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", name="bert_encoder")
    text_input = tf.keras.layers.Input(batch_size=32, shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])

    # Ajouter la dernière couche au modèle
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name="output")
    l = output_layer(l)

    # Charger les poids de la dernière couche depuis le fichier
    output_weights = np.load(weights_path, allow_pickle=True)
    output_layer.set_weights(output_weights)

    model = tf.keras.Model(inputs=[text_input], outputs=[l])

    return model


# Evaluation

def step_3_0_visual_evaluation(y_test, y_proba, y_pred, name_modele):
    """
    Cette fonction renvoie la matrice de confusion et la courbe ROC pour un modèle
    de classification binaire afin d'en juger les performances.


    :param y_test: labels des données de test
    :param y_proba: proba de prédiction de label
    :param y_pred: labels prédis
    :param name_modele: fin du nom propre au modele evalue
    """
    # Vider le répertoire de sortie
    if not os.path.exists(path_image):
        os.makedirs(path_image)

    # Affichage de la matrice de confusion
    name_matrice_conf = "{}/Mat_conf_{}_{}.png".format(path_image, formatted_date, name_modele)
    cm = confusion_matrix(y_test, y_pred)
    precision = cm / cm.sum(axis=0)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(precision, cmap='Blues')
    ax.set_xticks(range(len(precision)))
    ax.set_yticks(range(len(precision)))
    ax.set_xticklabels(["avis négatif", "avis positif"], fontsize=12)
    ax.set_yticklabels(["avis négatif", "avis positif"], fontsize=12)
    for i in range(len(precision)):
        for j in range(len(precision)):
            abs_value = cm[i][j]
            pct_value = cm[i][j] / cm[i].sum() * 100
            text = '{:.1f}%\n{}'.format(pct_value, abs_value)
            ax.annotate(text, xy=(j, i), ha='center', va='center', color='black',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7), fontsize=12)
    plt.xlabel('Prédiction', fontsize=18)
    plt.ylabel('Réalité', fontsize=18)
    plt.title('Matrice de confusion', fontsize=30, color='#2471a3')
    ax.title.set_position([.5, 200])
    plt.colorbar(im)
    plt.savefig(name_matrice_conf)
    mlflow.log_artifact(name_matrice_conf)
    plt.close()

    # Affichage de la courbe ROC
    name_courbe_roc = "{}/Courbe_ROC_{}_{}.png".format(path_image, formatted_date, name_modele)
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=4)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs', fontsize=18)
    plt.ylabel('Taux de vrais positifs', fontsize=18)
    plt.title('Courbe ROC (Receiver Operating Characteristic)', fontsize=22, color="#0d270f")
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(name_courbe_roc)
    mlflow.log_artifact(name_courbe_roc)
    plt.close()


def step_3_1_evaluation(y_test, y_pred, y_proba):
    """
    Cette fonction sert à évaluer les modèles.

    """
    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=4)
    roc_auc = auc(fpr, tpr)

    return accuracy, roc_auc


#######################
#      MODELES        #
#######################


def model_regression_logistique(file_path, embedding=0, sg=1, vect_size=100, min_count=1):
    """
    Cette fonction est une fonction assembleur qui lance de bout en bout l'entraînement et l'évaluation du
    modèle de regression logistique

    :param file_path: fichier CSV
    :param embedding: 0 ou 1, 0 pour utiliser la méthode d'embedding Word2Vec, 1 pour utiliser FastText
    :param sg: paramètre de Word2Vec
    :param vect_size: paramètre de Word2Vec
    :param min_count: paramètre de Word2Vec spécifiant le nombre minimum d'occurrences d'un mot pour être pris en compte

    """
    # initialisation MLOps
    fin_name = "{}_{}_{}_{}".format(embedding, sg, vect_size, min_count)
    mlflow.start_run(run_name="Regression Logistique {}".format(fin_name))

    bad_buzz = step_util_load_data(file_path)

    matrice_comm = None
    if embedding == 0:
        matrice_comm = step_0_1_word2vec(bad_buzz, sg, vect_size, min_count)
    else:
        matrice_comm = step_0_2_fasttext(bad_buzz, sg, vect_size, min_count)

    X_train, X_test, y_train, y_test = step_1_pca_train_test_split(bad_buzz['target'], matrice_comm)

    y_pred, y_proba = step_2_1_regression_logistique(X_train, X_test, y_train)
    y_proba = y_proba[:, 1]

    accuracy, roc_auc = step_3_1_evaluation(y_test, y_pred, y_proba)

    step_3_0_visual_evaluation(y_test, y_proba, y_pred, fin_name)

    # Partie mlflow
    mlflow.log_param("vector_size", vect_size)
    mlflow.log_param("sg", sg)
    mlflow.log_param("min_count", min_count)

    # Exemple de plusieurs métriques à enregistrer
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

    mlflow.log_metrics(metrics)
    mlflow.end_run()


def model_foret_aleatoire(file_path, embedding=0, sg=1, vect_size=100, min_count=1,
                          n_estimators=100, max_depth=5, max_samples=0.8):
    """
    Cette fonction est une fonction assembleur qui lance de bout en bout l'entraînement et l'évaluation du
    modèle de regression logistique

    :param file_path: fichier CSV
    :param embedding: 0 ou 1, 0 pour utiliser la méthode d'embedding Word2Vec, 1 pour utiliser FastText
    :param sg: paramètre de Word2Vec
    :param vect_size: paramètre de Word2Vec
    :param min_count: paramètre de Word2Vec spécifiant le nombre minimum d'occurrences d'un mot pour être pris en compte
    :param n_estimators: paramètre de RandomForestClassifier
    :param max_depth: paramètre de RandomForestClassifier
    :param max_samples: paramètre de RandomForestClassifier spécifiant la proportion d'échantillon à utiliser pour chaque arbre de decision

    """
    # initialisation MLOps
    fin_name = "{}_{}_{}_{}_{}_{}_{}".format(embedding, sg, vect_size, min_count, n_estimators, max_depth, max_samples)
    mlflow.start_run(run_name="Foret Aleatoire - {}".format(fin_name))

    bad_buzz = step_util_load_data(file_path)

    if embedding == 0:
        matrice_comm = step_0_1_word2vec(bad_buzz, sg, vect_size, min_count)
    else:
        matrice_comm = step_0_2_fasttext(bad_buzz, sg, vect_size, min_count)

    X_train, X_test, y_train, y_test = step_1_pca_train_test_split(bad_buzz['target'], matrice_comm)

    y_pred, y_proba = step_2_2_foret_aleatoire(X_train, X_test, y_train, n_estimators, max_depth, max_samples)
    y_proba = y_proba[:, 1]

    accuracy, roc_auc = step_3_1_evaluation(y_test, y_pred, y_proba)

    step_3_0_visual_evaluation(y_test, y_proba, y_pred, fin_name)

    # Partie mlflow
    mlflow.log_param("vector_size", vect_size)
    mlflow.log_param("sg", sg)
    mlflow.log_param("min_count", min_count)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_samples", max_samples)

    # Exemple de plusieurs métriques à enregistrer
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

    mlflow.log_metrics(metrics)
    mlflow.end_run()


def model_gradient_boosting_tree(file_path, embedding=0, sg=1, vect_size=100, min_count=1,
                                 n_estimators=100, max_depth=5, subsample=0.8):
    """
    Cette fonction est une fonction assembleur qui lance de bout en bout l'entraînement et l'évaluation du
    modèle de regression logistique

    :param file_path: fichier CSV
    :param embedding: 0 ou 1, 0 pour utiliser la méthode d'embedding Word2Vec, 1 pour utiliser FastText
    :param sg: paramètre de Word2Vec
    :param vect_size: paramètre de Word2Vec
    :param min_count: paramètre de Word2Vec spécifiant le nombre minimum d'occurrences d'un mot pour être pris en compte
    :param n_estimators: paramètre de RandomForestClassifier
    :param max_depth: paramètre de RandomForestClassifier
    :param subsample: paramètre de RandomForestClassifier spécifiant la proportion d'échantillon à utiliser pour chaque arbre de decision

    """
    # initialisation MLOps
    fin_name = "{}_{}_{}_{}_{}_{}_{}".format(embedding, sg, vect_size, min_count, n_estimators, max_depth, subsample)
    mlflow.start_run(run_name="Gradient boosting tree - {}".format(fin_name))

    bad_buzz = step_util_load_data(file_path)

    if embedding == 0:
        matrice_comm = step_0_1_word2vec(bad_buzz, sg, vect_size, min_count)
    else:
        matrice_comm = step_0_2_fasttext(bad_buzz, sg, vect_size, min_count)

    X_train, X_test, y_train, y_test = step_1_pca_train_test_split(bad_buzz['target'], matrice_comm)

    y_pred, y_proba = step_2_3_gradient_boosting_tree(X_train, X_test, y_train, n_estimators, max_depth, subsample)
    y_proba = y_proba[:, 1]

    accuracy, roc_auc = step_3_1_evaluation(y_test, y_pred, y_proba)

    step_3_0_visual_evaluation(y_test, y_proba, y_pred, fin_name)

    # Partie mlflow
    mlflow.log_param("vector_size", vect_size)
    mlflow.log_param("sg", sg)
    mlflow.log_param("min_count", min_count)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("subsample", subsample)

    # métriques à enregistrer
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

    mlflow.log_metrics(metrics)
    mlflow.end_run()


def model_bert(file_path):
    """
    Cette fonction prend en entrée un jeu de donnée qui servira pour l'evaluation d'un modèle bert déjà entrainé.
    Elle enregistre également les performances du modèle dans le mlflow.

    :param file_path: fichier CSV
    """

    mlflow.start_run(run_name="Modele Bert v2")

    # Chargment des données
    bad_buzz = step_util_load_data(file_path)

    # Chargement du modèle
    model = step_2_4_chargement_model_bert()

    y_proba = model.predict(bad_buzz.text)
    y_pred = np.vectorize(lambda x: 4 if x >= 0.5 else 0)(y_proba)
    y_test = bad_buzz.target

    accuracy, roc_auc = step_3_1_evaluation(y_test, y_pred, y_proba)

    fin_name = "modele_bert"
    step_3_0_visual_evaluation(y_test, y_proba, y_pred, fin_name)
    mlflow.tensorflow.log_model(model, 'model_bert')
    mlflow.tensorflow.save_model(model, "mlruns/models/Bert_Model")

    # métriques à enregistrer
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

    mlflow.log_metrics(metrics)
    mlflow.end_run()
