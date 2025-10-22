import streamlit as st
import pandas as pd

# Titulos de la página:
st.set_page_config(layout='centered', page_title='Mi primer app en streamlit', page_icon=":smiley:")

# Columnas de la página:
t1, t2, = st.columns([0.3,0.7])

# body:
st.write('Hola Mundo')

t1.image()

