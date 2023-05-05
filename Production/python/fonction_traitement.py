# Script contenant les fonctions utiles pour appliquer le nettoyage et formatage des données textuelles

#######################
#       Imports       #
#######################

import os
import re
import contractions
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from python.constantes import sample_abbr, emoticons_dict, stop_words


#######################
#      Fonctions      #
#######################


def clean_text(text):
    """
    Cette fonction prend un commentaire en paramètre et le renvoie normalisé
    :param text: commentaire sous forme de chaîne de caractères
    :return: commentaire normalisé
    """
    # On met tout en minuscules
    text = text.lower()

    # On supprime les formes contractés du texte pour uniformiser le format du texte avec la forme décontractée
    text = contractions.fix(text)

    # Remplacement des smiley par du texte
    text = " ".join([emoticons_dict.get(word, word) for word in text.split()])

    # Remplacement des abbréviations du texte
    text = " ".join([sample_abbr.get(word, word) for word in text.split()])

    # Remplacement de certaines parties du texte tels que les utilisateurs, les liens, les mentions
    text = re.sub(r'@\w+', 'AT_USER', text)
    text = re.sub(r'https?://\S+', 'URL', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    text = re.sub(r'[\s]+', ' ', text)
    text = re.sub(r'[^a-zA-Z_]', ' ', text)

    # Remplacement des suites de caractères répétées
    text = re.compile(r"(.)\1{2,}").sub(r"\1\1", text)

    # Correction des fautes d'orthographes
    text = ''.join(TextBlob(text).correct())

    return text


def lemmatize_text(text):
    """
    Cette fonction lemmatise le texte
    :param text: commentaire
    :return: commentaire lemmatisé
    """
    # Lemmatise
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]


def normalisation_texte(data):
    """
    Cette fonction prend en entrée un DataFrame pandas contenant au moins les colonnes target et text et renvoie
    Un DataFrame nettoyé et normalisé.

    :param data: DataFrame pandas contenant les commentaires et leur label
    :return: DataFrame pandas avec les commentaires nettoyés
    """

    data = data.apply(clean_text) \
        .apply(word_tokenize) \
        .apply(lemmatize_text) \
        .apply(lambda comm: [word for word in comm if word not in stop_words]) \
        .apply(lambda x: ["COMMENTAIRE_INSIGNIFIANT"] if x == [] else x) \
        .apply(lambda x: ' '.join(x))

    return data


def model_bert():
    """
    Cette fonction charge le modèle de classication bert préalablement entraîné

    :return: modèle bert
    """
    source = os.getcwd()
    model_path = "{}/Production/model_bert/poids_bert_class.h5".format(source)
    # Créer le modèle BERT
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", name="bert_preprocess")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", name="bert_encoder")
    text_input = tf.keras.layers.Input(batch_size=32, shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
    model = tf.keras.Model(inputs=[text_input], outputs=[l])

    new_model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs)
    new_model.load_weights(model_path)
    return new_model



