import streamlit as st
import numpy as np
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Trabalho 2 - Inteligência Artificial')
st.header('Reconhecimento de caracteres')

menu = st.sidebar

menu.file_uploader('Coloque suas fotos aqui', type=['png', 'jpg', 'jpeg'])
