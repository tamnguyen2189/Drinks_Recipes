import streamlit as st
import pandas as pd
import glob
import re
import numpy as np

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



st.title('Drinks Ingredients & Recipe')
st.markdown('Just for you')
st.image('images/european-specialty-coffee-market-by-pointbleu-design-blog.jpg', use_column_width ='always')

# Load data
path = r'C:/Users/pc/Desktop/my_git/final_project/recipes/clean_recipe.csv'
frame = pd.read_csv(path, index_col=None, header=0, encoding='utf-8')
df = frame.copy()

##########

##########

##########
# Searching ingredients in recipe:
def searching_recipe(ingredients, data):
    """ Dataset: df_clean
        Input: Ingredients
        Flow: searching recipe in dataset
        Output: recipes that contain ingredients
    """
    recipe_list = []
    for recipe in data:
        check = all(item in recipe for item in ingredients)
        if check == True:
            recipe_list.append(recipe)
    return recipe_list

########
col1, col2 = st.beta_columns(2)

with col1:
    # Enter ingredients
    name = st.text_input('Please enter ingredients here: (1-5 ingredients)')
    ingredients = name.split(' ')

with col2:
    # Select recipe:
    if len(ingredients[0]) > 0:
        recipe_opt = st.multiselect('Looking for familiar drinks, please select recipes you like:', searching_recipe(ingredients,df['recipe']))
    else:
        recipe_opt = st.multiselect('Please choose the recipe you like:',['Recipe not found'])


# Load Doc2Vec trained model:
model_path = 'C:/Users/pc/Desktop/my_git/final_project/model/doc2vecmodel_new.mod'
model = Doc2Vec.load(model_path) 

# Check similar vector
if len(recipe_opt) == 0:
    # Display recipes have found
    st.markdown('** With your ingredients you can find: **')
    if len(ingredients[0]) > 0:
        recipe_list = searching_recipe(ingredients,df['recipe'])
        if len(recipe_list)!=0:
            st.write(len(recipe_list),'`recipes`')
            # st.write(f' with your {len(ingredients)} ingredients')
            for i in range(len(recipe_list)):
                st.text(f'Recipe {i+1}: {recipe_list[i]}')
        else:
            st.write('Recipe not found')
    else:
        st.warning('You have to input at least one ingredient')
else:
    st.markdown('** Recommendation for similar drinks recipes **')
    test_doc = word_tokenize(recipe_opt[0])
    test_doc_vector = model.infer_vector(test_doc)
    similar_vetor = model.docvecs.most_similar(positive = [test_doc_vector])
    

    # col3, col4, col5 = st.beta_columns(3)
    # selection = []
    # with col3:
    #     for i in range(len(similar_vetor)//3):
    #         st.image(df['url_of_image'][similar_vetor[i][0]])
    #         # select_recipe = st.button(df['drink_name'][similar_vetor[i][0]])
    #         name = '[' + df['drink_name'][similar_vetor[i][0]] + ']'
    #         link = '(' + df['recipe_url'][similar_vetor[i][0]] + ')'
    #         st.markdown(name+link, unsafe_allow_html=True)
    #         st.text(df['recipe'][similar_vetor[i][0]])
    # with col4:
    #     for i in range(len(similar_vetor)//3,round(len(similar_vetor)*2/3)):
    #         st.image(df['url_of_image'][similar_vetor[i][0]])
    #         # select_recipe = st.button(df['drink_name'][similar_vetor[i][0]])
    #         # if select_recipe:
    #         name = '[' + df['drink_name'][similar_vetor[i][0]] + ']'
    #         link = '(' + df['recipe_url'][similar_vetor[i][0]] + ')'
    #         st.markdown(name+link, unsafe_allow_html=True)
    #         st.text(df['recipe'][similar_vetor[i][0]])
    # with col5:
    #     for i in range(round(len(similar_vetor)*2/3),len(similar_vetor)):
    #         st.image(df['url_of_image'][similar_vetor[i][0]])
    #         # select_recipe = st.button(df['drink_name'][similar_vetor[i][0]])
    #         name = '[' + df['drink_name'][similar_vetor[i][0]] + ']'
    #         link = '(' + df['recipe_url'][similar_vetor[i][0]] + ')'
    #         st.markdown(name+link, unsafe_allow_html=True)
    #         st.text(df['recipe'][similar_vetor[i][0]])

    # Recommendation:
    i = 0
    while i < len(similar_vetor):
        for _ in range(len(similar_vetor)-1):
            col = st.beta_columns(3)
            for num in range(3):
                if i < len(similar_vetor):
                    name = '[' + df['drink_name'][similar_vetor[i][0]] + ']'
                    link = '(' + df['recipe_url'][similar_vetor[i][0]] + ')'
                    col[num].image(df['url_of_image'][similar_vetor[i][0]])
                    col[num].markdown(name+link, unsafe_allow_html=True)
                    col[num].text(df['recipe'][similar_vetor[i][0]])
                i += 1
       
        