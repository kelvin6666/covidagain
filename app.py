import streamlit as st
st.set_page_config(layout="wide")
from multiapp import MultiApp
from apps import  eda, fs,ca

app = MultiApp()
app.add_app("Exploratory Data Analysis", eda.app)
app.add_app("Clustering Analysis", ca.app)
app.add_app("Feature Selection & Predictive Modelling", fs.app)


app.run()