import streamlit as st
import pandas as pd
import numpy as np
import os
from python.fonction_traitement import normalisation_texte, model_bert

st.title('Classification de commentaires')

#######################
#      Variables      #
#######################

source = os.getcwd()
MODEL = "{}/Production/bert_class".format(source)
model_prod = model_bert()

uploaded_file = st.file_uploader("Déposez un fichier csv avec des avis en anglais pour des restaurants \n"
                                 "(sur une colonne)")

if uploaded_file:
    df = pd.read_csv(uploaded_file,
                     squeeze=True,
                     names=["text"],
                     on_bad_lines='skip', encoding_errors='ignore',
                     encoding="ISO-8859-1")

    data_clean = normalisation_texte(df)
    res = model_prod.predict(data_clean)

    prediction = pd.DataFrame(df)
    prediction["Seuil_positif=0.4"] = np.vectorize(lambda x: "positif" if x >= 0.4 else "negatif")(res)
    prediction["Seuil_positif=0.5"] = np.vectorize(lambda x: "positif" if x >= 0.5 else "negatif")(res)
    prediction["Seuil_positif=0.6"] = np.vectorize(lambda x: "positif" if x >= 0.6 else "negatif")(res)

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

    st.write('prédiction en fonction du "seuil de coupure"')
    st.write(prediction)

