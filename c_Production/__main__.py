import streamlit as st
import pandas as pd
import numpy as np
import os
import emoji
from python.fonction_traitement import normalisation_texte, model_bert, pie_chart, \
    load_data, get_prediction, transform_comments_to_vectors, chargement_modeles_foret_aleatoire
from PIL import Image


##############################################
#      Chargement variables et modèles       #
##############################################

source = os.getcwd().replace(r"\\", "/")
WORD2VEC_PATH = "{}/b_Analyse/gestion_modeles/mlruns/models/Word2Vec/word2vec.wordvectors".format(source)

path_rf_image1 = "{}/b_Analyse/gestion_modeles/mlruns/0/1166531aed2d4a698a1edc9b673381dc/artifacts/Courbe_ROC_20230621232643_0_1_150_5_200_18_0.8.png".format(source)
path_rf_image2 = "{}/b_Analyse/gestion_modeles/mlruns/0/1166531aed2d4a698a1edc9b673381dc/artifacts/Mat_conf_20230621232643_0_1_150_5_200_18_0.8.png".format(source)
path_bert_image1 = "{}/b_Analyse/gestion_modeles/mlruns/0/fae0e2f9a2184d7ca16d9d163392269b/artifacts/Courbe_ROC_20230616185409_modele_bert.png".format(source)
path_bert_image2 = "{}/b_Analyse/gestion_modeles/mlruns/0/fae0e2f9a2184d7ca16d9d163392269b/artifacts/Mat_conf_20230616185409_modele_bert.png".format(source)

rf_image1 = Image.open(path_rf_image1)
rf_image2 = Image.open(path_rf_image2)
bert_image1 = Image.open(path_bert_image1)
bert_image2 = Image.open(path_bert_image2)

model_prod = model_bert()
scaler, pca, random_forest = chargement_modeles_foret_aleatoire(source)


########################
#      Traitement      #
########################

st.divider()
st.title("*Air Paradis* :sunglasses: : outil de prédiction de sentiment associé à un tweet.", )
st.header("Détectez les Bad Buzz grâce au Deep Learning")

st.write("Ici 2 modèles sont utilisés à chaque fois afin de pouvoir comparé les résultats. "
         "Un modèle bert basé sur des réseaux de neurones, et un modèle de forêt aléatoire associé à un modèle "
         "Word2Vec et un PCA afin de préparer les données au modèle.")

st.markdown("### Voici les performances théoriques des modèles :")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Forêt Aleatoire \n Accuracy : 73,8 %")
    st.image(rf_image1)
    st.image(rf_image2, caption='Forêt aléatoire, accuracy = 73.8 %')


with col2:
    st.markdown("#### Modèle Bert \n Accuracy : 73,5 %")
    st.image(bert_image1)
    st.image(bert_image2, caption='Modèle Bert, accuracy = 73.5 %')

st.divider()
st.divider()

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
        res_bert = model_prod.predict(data_clean)

        dt_clean_w2v = transform_comments_to_vectors(data_clean, WORD2VEC_PATH)
        dt_clean_w2v = scaler.transform(dt_clean_w2v)
        dt_clean_w2v = pca.transform(dt_clean_w2v)

        res_rf = random_forest.predict(dt_clean_w2v)

        prediction = pd.DataFrame(df)
        positif = emoji.emojize(':smiling_face_with_smiling_eyes:')
        negatif = emoji.emojize(':weary_face:')
        prediction["Anticipation du sentiment du tweet \n modèle Bert"] = np.vectorize(lambda x: positif if x >= 0.5 else negatif)(res_bert)
        prediction["Anticipation du sentiment du tweet \n Forêt Aléatoire"] = np.vectorize(lambda x: positif if x == 4 else negatif)(res_rf)

        prediction.set_index("tweet", inplace=True)
        st.table(prediction)


else:

    uploaded_file = st.file_uploader("Déposez un fichier csv avec des avis en anglais. \n"
                                     "(sur une colonne)")

    if uploaded_file:
        df = load_data(uploaded_file)

        data_clean = normalisation_texte(df)
        res_bert = get_prediction(model_prod, data_clean)

        dt_clean_w2v = transform_comments_to_vectors(data_clean, WORD2VEC_PATH)
        dt_clean_w2v = scaler.transform(dt_clean_w2v)
        dt_clean_w2v = pca.transform(dt_clean_w2v)

        res_rf = random_forest.predict(dt_clean_w2v)
        proba_rf = random_forest.predict_proba(dt_clean_w2v)[:, 1]

        prediction = pd.DataFrame(df)
        positif = emoji.emojize(':smiling_face_with_smiling_eyes:')
        negatif = emoji.emojize(':weary_face:')

        # Initialisation du seuil de coupure à 0.5
        seuil_coupure = st.slider("Sélectionnez un seuil de coupure", 0.3, 0.8, 0.5, 0.02)

        prediction["Anticipation du sentiment du tweet \n modèle Bert"] = np.vectorize(lambda x: positif if x >= seuil_coupure else negatif)(res_bert)
        prediction["Anticipation du sentiment du tweet \n Forêt Aléatoire"] = np.vectorize(lambda x: positif if x >= seuil_coupure else negatif)(proba_rf)

        prediction.set_index("tweet", inplace=True)
        st.dataframe(prediction)

        st.header("Répartition entre les commentaires")
        st.subheader("Seuil de coupure = {}".format(seuil_coupure))

        col1, col2 = st.columns(2)
        with col1:
            st.write("Modèle Bert")
            pie_chart(res_bert, seuil_coupure)

        with col2:
            st.write("Modèle Forêt Aléatoire")
            pie_chart(proba_rf, seuil_coupure)


st.divider()

