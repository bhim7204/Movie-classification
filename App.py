import streamlit as st
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import plotly.express as px
MAX_SEQUENCE_LENGTH = 10000

loaded_model = load_model('MovieClassify.h5')
classes = [ "Horror",  "Action",  "Documentary", "Science Fiction"]

with (open("tokenizer.pkl", "rb")) as openfile:
        # while True:
        try:
            tokenizer = pickle.load(openfile)
        except EOFError:
            print("Err")

def build_url(params, pages=1):
    url= "https://api.themoviedb.org/3/search/movie?api_key=50298e4210ad98f9daba9b5d76b15889&language=en-US&include_adult=false"
    # url = "https://api.themoviedb.org/3/discover/movie?api_key=50298e4210ad98f9daba9b5d76b15889&language=en-US&sort_by=popularity.desc&include_adult=false&include_video=false&with_watch_monetization_types=flatrate"
    url += f"&query={params}&page={pages}"
    return url
def get_movie_data(movie_name, pages=1):
    response = requests.get(build_url(movie_name, pages))
    return response.json()

st.set_page_config(
    page_title="Movie Classification",
    page_icon="U+1F3AC",
    # layout="wide",
    # initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'vigneshkkar@gmail.com',
    #     'Report a bug': "vigneshkkar@gmail.com",
    #     'About': "# This is a app to classify the movie Genere using the plot"
    # }
)

st.title('Movie Classification')

st.subheader('Enter Movie name to detect the Genre')

title = st.text_input('Movie title', 'Matrix')
if st.button('Predict the Movie Genre'):
     for i in get_movie_data(title)["results"]:
        with st.expander(i['title']):
            predicted = loaded_model.predict(pad_sequences( tokenizer.texts_to_sequences([i['overview']]), maxlen = MAX_SEQUENCE_LENGTH))
            fig = px.line_polar(r=predicted[0], theta=classes, line_close=True, start_angle=30,)
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)
            st.header("Movie Name:")
            st.write(i['title'])
            st.header("Description/Plot:")
            st.write(i['overview'])
            st.header("Poster:")
            if i["poster_path"]:
                st.image("https://image.tmdb.org/t/p/original" + i["poster_path"])
            else:
                st.write("No Poster Found")
            # st.write(predicted)

else:
     st.write('')