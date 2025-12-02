import os
import pandas as pd
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
LOCAL_PATH = "data/pima-indians-diabetes.csv"

COL_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

def ensure_data():
    """Download dataset apabila belum ada di folder data/"""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(LOCAL_PATH):
        try:
            df = pd.read_csv(DATA_URL, header=None)
            df.columns = COL_NAMES
            df.to_csv(LOCAL_PATH, index=False)
            return df
        except Exception as e:
            st.error(f"Gagal mendownload dataset: {e}")
            raise
    else:
        df = pd.read_csv(LOCAL_PATH)
        df.columns = COL_NAMES
        return df

def load_local(path):
    df = pd.read_csv(path)
    return df
