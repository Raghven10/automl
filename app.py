import os

import pandas as pd
import streamlit as st

from ydata_profiling import ProfileReport

from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import setup, compare_models, pull, save_model


with st.sidebar:
    st.image("ml_img.jpg")
    st.title("Auto Stream ML")
    choice = st.radio("Navigation", ["Upload", "Train"])
    st.info("An automated ML pipeline app using PYdata")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload your dataset for Modelling")
    file = st.file_uploader("Upload your file here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
        if st.button("generate report"):
            profile_report = ProfileReport(df, title="Profiling Report")
            st_profile_report = st_profile_report(profile_report)


if choice == "ML":
    st.title("Select Target and Train Model")
    target = st.selectbox("Select Target", df.columns)
    if st.button("Train Model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("Analysis of data to train the model on selected target")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("Here is the trained ml model")
        st.dataframe(compare_df)
        save_model(best_model, "best_model")

        with open("best_model.pkl", "rb") as f:
            st.download_button("Download model", f, "best_model.pkl")


if choice == "Download":
    st.title("Download the trained Model")
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the model", f, "best_model.pkl")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
