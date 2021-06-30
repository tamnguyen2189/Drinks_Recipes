import streamlit as st
import pandas as pd
import glob
import re
import numpy as np

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
from gensim.models import Phrases
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess
import pickle
from pprint import pprint
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.neighbors import NearestNeighbors


st.title('Your Recipe')
st.markdown('Just for you')
st.image('images/european-specialty-coffee-market-by-pointbleu-design-blog.jpg', use_column_width ='always')

# Load data
# path = r'C:/Users/pc/Desktop/my_git/final_project/recipes/clean_recipe.csv'
path = r'tamnguyen2189/drinks_recipes/master/clean_recipe.csv'
frame = pd.read_csv(path, index_col=None, header=0, encoding='utf-8')
df = frame.copy()

##########
# Load LDA model, corpus, dictionary:
# lda_model_path = 'C:/Users/pc/Desktop/my_git/final_project/model/lda_5.model'
# corpus_path = 'C:/Users/pc/Desktop/my_git/final_project/model/corpus.pkl'
# dictionary_path = 'C:/Users/pc/Desktop/my_git/final_project/model/dict.pkl'
# nbrs_path = 'C:/Users/pc/Desktop/my_git/final_project/model/nbrs_5.pkl'

lda_model_path = 'tamnguyen2189/drinks_recipes/master/model/lda_5.model'
corpus_path = 'tamnguyen2189/drinks_recipes/master/model/corpus.pkl'
dictionary_path = 'tamnguyen2189/drinks_recipes/master/model/dict.pkl'
nbrs_path = 'tamnguyen2189/drinks_recipes/master/model/nbrs_5.pkl'

lda_model = gensim.models.ldamodel.LdaModel.load(lda_model_path)
corpus = pickle.load(open(corpus_path, 'rb'))
dictionary = pickle.load(open(dictionary_path, 'rb'))
nbrs = pickle.load(open(nbrs_path, 'rb'))

# Load Doc2Vec trained model:
model_path = 'tamnguyen2189/drinks_recipes/master/model/doc2vecmodel_final.mod'
model = Doc2Vec.load(model_path) 

##########
# Searching ingredients in recipe:

def searching_recipe(ingredients,data):
    """ Dataset: df_clean
        Input: Ingredients
        Flow: searching recipe in dataset
        Output: recipes that contain ingredients
    """
    recipe_list = []
    drink_name = []
    recipe_url = []
    image_url = []
    for i in range(len(data['recipe'])):
        check_recipe = all(item in data['recipe'][i] for item in ingredients)
        check_name = all(item in data['drink_name'][i] for item in ingredients)
        if check_recipe == True:
            recipe_list.append(data['recipe'][i])
            drink_name.append(data['drink_name'][i])
            recipe_url.append(data['recipe_url'][i])
            image_url.append(data['url_of_image'][i])
        elif check_name == True:
            recipe_list.append(data['recipe'][i])
            drink_name.append(data['drink_name'][i])
            recipe_url.append(data['recipe_url'][i])
            image_url.append(data['url_of_image'][i])
    return recipe_list, drink_name, recipe_url, image_url

# Create stop_words:
stop_words = stopwords.words('english')
stop_words.extend(['tablespoons', 'tablespoon','cup', 'cups', 'ounce', 'ounces','teaspoon','teaspoons','coarse','grind',
                   'kosher','sea','zest', 'ground', 'extract', 'frozen', 'bottle', 'whole', 'taste','fresh', 'white',
                   'fluid','powder','sauce','syrup','large','small','chopped','granulated', 'cubes', 'concentrate', 
                   'wedge', 'flour', 'wedge', 'club', 'inch', 'dry', 'medium','red', 'whipped', 'yellow', 'milliliter', 
                   'triple', 'sec', 'optional', 'light', 'simple', 'slice', 'gram',  'instant',  'sliced',  'brown',
                   'dark', 'heavy',  'peeled', 'chilled', 'stick', 'cut', 'sticks', 'dried', 'half', 'black', 'twist', 'green'])

# Lemmatization fuction:
lemmatizer = WordNetLemmatizer()

def lemma_stop(row):
    return ' '.join([lemmatizer.lemmatize(word) for word in row.split() if word not in stop_words])

def lemma_stop_input(row):
    return ' '.join([lemmatizer.lemmatize(word) for word in row if word not in stop_words])

# Remove stop_words:
df['no_stop'] = df['recipe'].apply(lemma_stop)

# Document:
text_array = df['no_stop']

# Phrase modeling: Bi-grams and Tri-grams
def docs_with_grams(docs):
    """ Input a list of sentences.
        Output the list of sentences including bigram and trigram.
    """
    docs = np.array(list(map(lambda x: x.split(), docs)))
    bigram = Phrases(docs, min_count=10)
    trigram = Phrases(bigram[docs])

    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
        for token in trigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs

# Function to lemmatize & remove stop_words of test doc:
def preprocess_text(test_array):
    ''' Preprocess input text
        Output: list of vector of input text '''
    # Lemmatize and remove stop words    
    test_array = [lemma_stop_input(test_array)]
    # List of sentence include bigram and trigram
    docs = docs_with_grams(test_array)
    # Get corpus
    test_corpus = [dictionary.doc2bow(text) for text in docs] #id2word
    return test_corpus

# Function to find Vector of test doc
def doc_vecs(test_array):
    test_corpus = preprocess_text(test_array)
    result_vecs = []
    for i in range(len(test_corpus)):
        top_topics = lda_model.get_document_topics(test_corpus[i], minimum_probability=0.0)
        topic_vec = list(map(lambda x:x[1], top_topics))
        result_vecs.append(topic_vec)
    return result_vecs

##########
# Topic vector in document:
recipe_vecs = []
for i in range(text_array.shape[0]):
    top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = list(map(lambda x:x[1], top_topics))
    recipe_vecs.append(topic_vec)

# Recipe vector & their topics:
topic_recipe = pd.DataFrame(recipe_vecs)
topics = np.argmax(recipe_vecs,axis=1)
topic_recipe['topics'] = topics

# Creat topic dictionary base on model
topic_dict = {0: 'fruit_juice',
              1: 'cocktail',
              2: 'liqueur',
              3: 'cream_milk_coffee',
              4: 'spice'}


##########
col1, col2 = st.beta_columns(2)

with col1:
    # Enter ingredients
    name = st.text_input('Please enter ingredients here: (1-5 ingredients)')
    ingredients = name.split(' ')

with col2:
    # Select recipe:
    if len(ingredients[0]) > 0:
        recipe_opt, drink_name,_,_ = searching_recipe(ingredients,df)
        opt = [f'{drink_name[i]}: {recipe_opt[i]}' for i in range(len(recipe_opt))]
        recipe_opt = st.multiselect('Looking for familiar drinks, please select recipes you like:', opt)
        topic = st.button('Looking for more wide ideas')       
    else:
        recipe_opt = st.multiselect('Please choose the recipe you like:',['Recipe not found'])
        topic = st.button('Looking for more wide ideas')

if topic == True:
    if len(ingredients[0]) > 0:
        # Input and preprocess test doc
        result_vecs = doc_vecs(ingredients)
        distances, indices = nbrs.kneighbors(result_vecs)
        # for i in indices[0]:
        #     name = '[' + df.iloc[i]['drink_name'] + ']'
        #     link = '(' + df.iloc[i]['recipe_url'] + ')'
        #     st.write('Topic: ' + topic_dict[topic_recipe['topics'][i]])
        #     st.markdown(name+link, unsafe_allow_html=True)
        #     st.text('Recipe: ' + df.iloc[i]['recipe'])
        i = 0
        while i < len(indices[0]):
            for _ in range(len(indices[0])-1):
                col = st.beta_columns(3)
                for num in range(3):
                    if i < len(indices[0]):
                        name = '[' + df.iloc[indices[0][i]]['drink_name'] + ']'
                        link = '(' + df.iloc[indices[0][i]]['recipe_url'] + ')'
                        col[num].image(df.iloc[indices[0][i]]['url_of_image'])
                        col[num].markdown(name+link, unsafe_allow_html=True)
                        col[num].write('Topic: ' + topic_dict[topic_recipe['topics'][indices[0][i]]])
                        col[num].text('Recipe: ' + df.iloc[indices[0][i]]['recipe'])
                    i += 1

    else:
        st.warning('You have to input at least one ingredient')

else:

    
    # Check similar vector
    if len(recipe_opt) == 0:
        # Display recipes have found
        st.markdown('** With your ingredients you can find: **')
        if len(ingredients[0]) > 0:
            recipe_list, drink_name, recipe_url, image_url = searching_recipe(ingredients,df)
            
            if len(recipe_list)!=0:
                st.write(len(recipe_list),'`recipes`')
                for i in range(len(recipe_list)):
                    # st.image(image_url[i])
                    link = '[' + drink_name[i] + ']' + '(' + recipe_url[i] + ')'
                    st.markdown(link, unsafe_allow_html=True)
                    st.text(f'Recipe: {recipe_list[i]}')

            else:
                st.write('Recipe not found')
        else:
            st.warning('You have to input at least one ingredient')
    elif len(recipe_opt) != 0:
        st.markdown('** Recommendation for similar drinks recipes **')
        test_doc = word_tokenize(recipe_opt[0])
        test_doc_vector = model.infer_vector(test_doc)
        similar_vetor = model.docvecs.most_similar(positive = [test_doc_vector])
        

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
                        col[num].text('Recipe: ' + df['recipe'][similar_vetor[i][0]])
                    i += 1


    








