import streamlit as st

st.set_page_config(
    page_title="Wine Quality ML",
    page_icon="🍷",
)

st.write("# Welcome to Wine Quality ML! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    The dataset for this competition (both train and test) was generated from a deep learning model trained on the Wine Quality dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.
    **👈 Select a page from the sidebar** to see some feataures
    of what Wine Quality ML can show!
    ### Want to learn more?
    - Check out original [notebook](https://www.kaggle.com/code/lusfernandotorres/wine-quality-eda-prediction-and-deploy) by Luís Fernando Torres.
    - Jump into step-by-step [tutorial](https://ai.plainenglish.io/how-i-deployed-a-machine-learning-model-for-the-first-time-b82b9ea831e0) by Luís Fernando Torres.
"""
)