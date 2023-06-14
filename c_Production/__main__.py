import streamlit as st
import pandas as pd
import numpy as np
import os
import emoji
from python.fonction_traitement import normalisation_texte, model_bert, pie_chart, load_data, get_prediction

st.title("*Air Paradis* :sunglasses: : outil de prédiction de sentiment associé à un tweet.", )
st.header("Détectez les Bad Buzz grâce au Deep Learning")

#######################
#      Variables      #
#######################

source = os.getcwd()
MODEL = "{}/c_Production/bert_class".format(source)
model_prod = model_bert()

# Définir les options pour la sélection
option_1 = '1 seul'
option_2 = "Plus d'un"
options = (option_1, option_2)

# Afficher la sélection sous forme de boutons radio
choix = st.radio("Pour combien de Tweet voulez-vous prédire le sentiment associé ?", options)

# Afficher le sentiment associé à un tweet ou plus en fonction du choix de l'utilisateur.
if choix == option_1:
    avis = st.text_input("Ecrivez ici votre avis (en anglais) : ")

    if avis:
        df = pd.Series([avis], name="tweet")
        data_clean = normalisation_texte(df)
        res = model_prod.predict(data_clean)

        prediction = pd.DataFrame(df)
        positif = emoji.emojize(':smiling_face_with_smiling_eyes:')
        negatif = emoji.emojize(':weary_face:')
        prediction["Anticipation du sentiment du tweet"] = np.vectorize(lambda x: positif if x >= 0.5 else negatif)(res)

        prediction.set_index("tweet", inplace=True)
        st.table(prediction)


else:

    uploaded_file = st.file_uploader("Déposez un fichier csv avec des avis en anglais. \n"
                                     "(sur une colonne)")

    if uploaded_file:
        df = load_data(uploaded_file)

        data_clean = normalisation_texte(df)
        res = get_prediction(model_prod, data_clean)

        prediction = pd.DataFrame(df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Commentaires innitiaux :")
            st.write(df)

        with col2:
            st.write("Commentaires normalisés :")
            st.write(data_clean)

        with col3:
            st.write("Probabilité de classification :")
            st.write(res)

        # Initialisation du seuil de coupure à 0.5
        seuil_coupure1, seuil_coupure2 = st.slider("Sélectionnez un seuil de coupure", 0.0, 1.0, (0.5, 0.6), 0.05)

        # Mise à jour de la prédiction en fonction du seuil de coupure sélectionné
        prediction[f"Seuil_positif={seuil_coupure1}"] = np.vectorize(
            lambda x: "positif" if x >= seuil_coupure1 else "negatif")(res)
        prediction[f"Seuil_positif={seuil_coupure2}"] = np.vectorize(
            lambda x: "positif" if x >= seuil_coupure2 else "negatif")(res)

        st.write('prédiction en fonction du "seuil de coupure"')
        st.write(prediction)

        col1, col2 = st.columns(2)
        st.write("Répartition entre les commentaires")

        with col1:
            st.write("Seuil de coupure = {}".format(seuil_coupure1))
            pie_chart(res, seuil_coupure1)

        with col2:
            st.write("Seuil de coupure = {}".format(seuil_coupure2))
            pie_chart(res, seuil_coupure2)


st.divider()


# Titre de la section "Comment ça marche"
st.header("Comment ça marche ?")

# Description du fonctionnement du modèle
st.write("Notre modèle de classification de commentaires utilise un algorithme de traitement "
         "du langage naturel pour analyser le contenu des commentaires et prédire leur polarité. "
         "Le modèle fonctionne en plusieurs étapes :")

# Étape 1 : Prétraitement du texte
st.subheader("Étape 1 : Prétraitement du texte")
st.write("Avant d'être analysé par le modèle, chaque commentaire subit un processus de prétraitement qui consiste "
         "à nettoyer et normaliser le texte. Les étapes de prétraitement comprennent entre autres :")
st.write("- Suppression de la ponctuation et des caractères spéciaux")
st.write("- Mise en minuscules des caractères")
st.write("- Suppression des stop words")
st.write("- Lemmatisation des mots")

# Étape 2 : Encodage des tokens
st.subheader("Étape 2 : Encodage des tokens")
st.write("Le modèle utilise ensuite une technique d'encodage des tokens pour représenter chaque "
         "commentaire sous forme de séquence de nombres. Cette étape permet de transformer le texte "
         "en données exploitables par le modèle.")
st.write("Nous avons utilisé le tokenizer de BERT pour encoder les commentaires.")

# Étape 3 : Classification des commentaires
st.subheader("Étape 3 : Classification des commentaires")
st.write("Le modèle utilise une approche de classification binaire pour prédire la polarité de chaque commentaire : "
         "positif ou négatif. Cette étape se base sur un modèle d'apprentissage automatique qui a été entraîné sur "
         "un ensemble de données annotées.")
st.write("Notre modèle est capable de prédire avec une grande précision la polarité de nouveaux commentaires.")


# Conclusion
st.write("Grâce à ce processus, notre modèle est capable de classifier les commentaires en fonction "
         "de leur polarité, ce qui peut être utile pour diverses applications, telles que l'analyse "
         "de sentiment, la détection de spam ou la surveillance de la réputation en ligne.")

st.divider()

