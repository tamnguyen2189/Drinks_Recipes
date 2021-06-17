import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

PATH = "media/AB_NYC_2019.csv"

# Create a side menu 
menu = ['Home', 'Doggy Reads Data', 'Doggy Reads Image']
choice = st.sidebar.selectbox('What can Doggy do?', menu)

# Create the Home page
if choice == 'Home':
    st.header('Doggy Wonderworld')
    st.image('media/isle_of_dog.gif', use_column_width="always")

    # Layout your content
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        name = st.text_input("What's doggy name?")
        if name: 
            st.write(name, " is a cute name!")
    with col2:
        old = st.slider('How old is doggy?')
    with col3:
        st.selectbox("What's doggy loves to eat?", ['Carrot', 'Bones', 'Chocolate'])

# Create the First page
elif choice == 'Doggy Reads Data':
    st.header('Hotdog Summer')
    st.image('media/9e1b49d166612f7a7846aa5b77b871c7.gif')

    # Load data
    @st.cache
    def load_data(path):
        return pd.read_csv(path)
    
    data = load_data(PATH)
    st.dataframe(data)

    # Create the bar chart
    col1, col2 = st.beta_columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 5))
        data.groupby('neighbourhood_group')['price'].mean().plot(kind='barh', ax=ax)
        st.pyplot(fig)
    with col2:
        st.write('We have all the money in this world. Manhattan we go.')

    # Create the world map
    price = st.slider('Choose your price', min_value=100, max_value=1000)
    new_data = data[data['price']>price]
    st.map(new_data[['latitude', 'longitude']])

# Create the Second page
elif choice == 'Doggy Reads Image':
    st.header('Doggy Reads Image')

    # Image uploader
    image_file = st.file_uploader("Choose Your Image", key=1)
    
    if image_file != None:
        # Show info
        file_details = {"Filename":image_file.name, 
                        "FileType":image_file.type,
                        "FileSize":image_file.size}

        st.write(file_details)

        # Get image
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, use_column_width="always")
        
        # Upload an audio
        audio_file = st.file_uploader("Add me a Woof Woof", key=2)
        if audio_file != None:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')


