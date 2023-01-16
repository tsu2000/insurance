import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

import io
import requests

from PIL import Image
from streamlit_extras.badges import badge

# Import sklearn methods/tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Import all sklearn algorithms used


def main():
    col1, col2, col3 = st.columns([0.05, 0.265, 0.035])
    
    with col1:
        url = 'https://github.com/tsu2000/insurance/raw/main/images/shield.png'
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        st.image(img, output_format = 'png')

    with col2:
        st.title('&nbsp; Insurance ML Application')

    with col3:
        badge(type = 'github', name = 'tsu2000/insurance', url = 'https://github.com/tsu2000/insurance')

    st.markdown('### 📋 &nbsp; Insurance Machine Learning Web App')
    st.markdown('This web application aims to explore various regression models')

    # Initialise dataframe
    url = 'https://raw.githubusercontent.com/tsu2000/insurance/main/.csv'
    df = pd.read_csv(url)

    st.dataframe(df)



def lr_model():
    pass

def rf_model():
    pass
    

if __name__ == "__main__":
    st.set_page_config(page_title = 'Insurance ML App', page_icon = '📋')
    main()