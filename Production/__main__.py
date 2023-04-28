import streamlit as st
import pandas as pd
from Production.python.pretraitement import normalisation_texte

st.title('Classification de commentaires')
st.write("Hello Word")

URL_CSV = "https://github.com/Alphonse-Raclon/P7_classification_commentaires/blob/develop/Production/ressources/commentaire.csv"


st.text("On va maintenant charger un dataframe composé de commentaires à propos de restaurants")

data = pd.read_csv(URL_CSV,
                   squeeze=True,
                   names=["text"],
                   on_bad_lines='skip', encoding_errors='ignore',
                   encoding="ISO-8859-1")

choice = st.slider("Choise un nombre de commentaires que tu veux afficher", 1, 20)
data_select = data.sample(choice)
st.dataframe(data_select)

st.text("On va maintenant prétraiter ces commentaires pour pouvoir les rentrer dans le modèle prédictif")
st.text("Traitement en cours ....")
data_clean = normalisation_texte(data_select)

st.text("Commentaires normalisés : ")
st.dataframe(data_clean)


