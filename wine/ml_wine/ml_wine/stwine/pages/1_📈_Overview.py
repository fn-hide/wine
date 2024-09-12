import time
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Overview", page_icon="📊")

st.markdown("# Overview")
st.sidebar.header("Overview")
st.write(
    """This page shows an overview of Wine Quality dataset. Enjoy!"""
)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return pd.read_csv('ml_wine/data/train.csv')


# status_text = st.sidebar.empty()
# progress_bar = st.sidebar.progress(0)

# status_text.text("%i%% Complete" % i)
# progress_bar.progress(i)

df = get_dataset()
df = df.drop(columns='Id')
n_samples, n_features = df.shape

col1, col2 = st.columns(2)
col1.metric(label="Samples", value=f"{n_samples}")
col2.metric(label="Features", value=f"{n_features}")

st.dataframe(df.head(), hide_index=True)

'''## Statistics'''
st.dataframe(df.describe().T)

