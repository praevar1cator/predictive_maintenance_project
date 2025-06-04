import streamlit as st

pages = {
    "Анализ и модель": "analysis_and_model.py",
    "Презентация": "presentation.py"
}

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти", list(pages.keys()))
page = pages[selection]

with open(page, encoding="utf-8") as f:
    exec(f.read())
