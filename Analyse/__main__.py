import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow

from gensim.models import Word2Vec



def main(bad_buzz):
    bad_buzz.text_clean = bad_buzz.text_clean.apply(lambda x: x.split(" "))

    # Créer une liste de phrases à partir de la colonne "text_clean" du dataframe
    sentences = bad_buzz.text_clean.tolist()

    # Entraîner le modèle Word2Vec avec la méthode CBOW
    model = Word2Vec(sentences, sg=0, vector_size=100, min_count=1, workers=8)
    # Combinaison des listes de mots en une seule liste
    all_words = [word for comment in bad_buzz['text_clean'] for word in comment]

    # Récupération des vecteurs pour tous les mots
    vectors = model.wv[all_words]

    # Création d'un dictionnaire pour accéder aux vecteurs par mot
    word_vectors = {word: vectors[i] for i, word in enumerate(all_words)}

    # Création d'une colonne 'vect' contenant les vecteurs de chaque commentaire
    bad_buzz['vect'] = bad_buzz['text_clean'].apply(
        lambda comment: np.mean([word_vectors[word] for word in comment if word in word_vectors], axis=0))

    # Préparation des données pour former un dataframe
    vect_comm = bad_buzz.vect.apply(pd.Series)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Standardisation des données
    scaler = StandardScaler()
    vect_comm_std = scaler.fit_transform(vect_comm)

    # Calcul des composantes principales
    n_comp = 0.99

    pca = PCA(n_components=n_comp)
    res_vect_comm = pca.fit_transform(vect_comm_std)


    X = res_vect_comm
    y = bad_buzz.target

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # Entraîner le modèle
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Faire des prédictions sur les données de test
    y_pred = clf.predict(X_test)

    # Évaluer les performances du modèle
    from sklearn.metrics import accuracy_score
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return bad_buzz.text_clean




