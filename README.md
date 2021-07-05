# Your_Recipe
Introduction
Barista and bartender have to constantly develop new recipes to refresh their menu. Therefore, this project will help them solve that problem.
Your Recipe is an app, where you can input your ingredients and it will return delicious recipes base on the ingredients you have. The app also suggest other drinks similar to the one you selected.
![Project Flow](/images/viber_image_2021-07-03_11-43-59.jpg)
# Project Implementation
1. Dataset
2. Reading and Cleaning Data
3. Searching recipe by pandas
4. Doc2Vec model
5. Latent Dirichlet Allocation model (LDA)
6. Nearest Neighbors
*****
## 1. Dataset

In this project, I will build a Doc2Vec model and a LDA topic model from a dataset that I scraped from The Spruce Eats and All Recipes. 
We will then use these model to find the most similar recipes to one we might like to use from the dataset, in order to recommend other things we could use.
![Dataset](/images/dataset.jpg)

## 2. Reading and cleaning Data
Reading data
```ruby
import pandas as pd
import numpy as np
import glob
import re

# Load data: concatenate all recipes into one file
path = r'/content/gdrive/MyDrive/Colab Notebooks/Final Project/Recipes' # use your path
files = glob.glob(path + "/*.csv")
all_files = []

for filename in files:
     df_sub = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8')
     all_files.append(df_sub)

frame = pd.concat(all_files, axis=0, ignore_index=True)
df = frame.copy()
```

Cleaning data
```ruby
# Drop wrong data
df.drop(df[df['recipe'] == '[]'].index, inplace= True)

# Drop duplicated data
df.drop_duplicates(subset ='recipe',
                     keep = 'first', inplace = True)
                     
# Drop all na values
df.dropna(axis=0, inplace=True)

df_clean = df.copy()
```

```ruby
# Function to clean recipes in dataset
def clean_recipe(recipe):
    pat1 = r'[\'\"\(\)*,:;.!?~\[\]\{\}]'
    pat2 = r'\\\w{2}\d'  
    pat3 = r'\\\w\d{4}'
    pat4 = r'\\\w\d{3}\w'
    clean_recipe = recipe.replace('-',' ').lower()  #.strip('-[]{}\'!?~')
    clean_recipe = re.sub(pat1,'',clean_recipe)
    clean_recipe = re.sub(pat2,'',clean_recipe)
    clean_recipe = re.sub(pat3,'',clean_recipe)
    clean_recipe = re.sub(pat4,'',clean_recipe)
    return clean_recipe
    
# Apply cleaning function on df_clean
df_clean['recipe'] = df_clean['recipe'].apply(lambda x: clean_recipe(x))
df_clean['drink_name'] = df_clean['drink_name'].apply(lambda x: clean_recipe(x))

# Drop recipe with error url
df_clean.drop(df_clean[df_clean['url_of_image']=='/img/icons/generic-recipe.svg'].index, inplace= True)
```
![Clean_data](/images/clean_data.jpg)
Saving data
```ruby
# Save data for training after cleaning
data = df_clean
path = '/content/gdrive/MyDrive/Colab Notebooks/Final Project/Recipes'
data.to_csv(path + "/clean_recipe_training.csv", index=False)
```
## 3. Searching recipe by pandas
At this part, I use pandas to find the recipe that have exactly all the input ingredient. Let define the function for searching:
```ruby
# Function to search recipes in dataset
def searching_recipe(ingredients, data):
    """ Dataset: df_clean
        Input: Ingredients
        Flow: searching recipe in dataset
        Output: all informations that contain ingredients
    """
    recipe_list = []
    drink_name = []
    recipe_url = []
    image_url = []
    for i in range(len(data['recipe'])):
        check_recipe = all(item in data['recipe'][i] for item in ingredients)
        check_name = any(item in data['drink_name'][i] for item in ingredients)
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
```
   
## 4. Doc2Vec model
There are multiple methods to change the text into vectors:
- Label Encoding
- Custom binary Encoding
- One-Hot Encoding
But these methods will lose the context of a given text. Doc2Vec can solve this problem by creating vectors out of a document, independent of the document length.

```ruby
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
```

Preprocessing data:
Getting training data
We define our list of sentences. Here is the column of recipe. It's good to use a list of sentences for easier processing of each sentence.
We will also keep a tokenized version of these sentences

```
# Find vector of chosen recipe
sentence = []
for i in df_clean['recipe']:
    sentence.append(i)

# Tokenization of each document
tokenized_sent = []
for s in sentence:
    tokenized_sent.append(word_tokenize(s))

# 
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
```
![Tagged_data](/images/tag_doc.jpg)
Training model

```ruby
# Train doc2vec model
model = Doc2Vec(tagged_data, vector_size = 50, window = 2, min_count = 2, epochs = 30)
```
Now, we give a test sentence. The infer_vector function returns the vectorized form of test sentence.

Let's call the most_similar function, which returns top most similar sentences and its indices throughout the document.

```
# Input test sentence
input = "2 ounces gin, 4 ounces tonic, strawberry"

# Change input sentence into vector
test_doc = word_tokenize(input.lower())
test_doc_vector = model.infer_vector(test_doc)

# Find similar vectors
model.docvecs.most_similar(positive = [test_doc_vector])
```
![Similar_vec](/images/similar_vec.jpg)

## 5. Latent Dirichlet Allocation model (LDA)
```ruby
# Import libraries
import gensim
from gensim.models import Phrases
from gensim.models.coherencemodel import CoherenceModel
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
```
```ruby
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

# Remove stop_words:
df_clean['no_stop'] = df_clean['recipe'].apply(lemma_stop)
```

```ruby
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
    
def get_dict_corpus(docs):
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=10, no_above=0.2)

    # Getting corpus 
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return dictionary, corpus
```
Data transformation:
Corpus and Dictionary
Bi-grams and Tri-grams
```ruby
# Phrase modeling: Bi-grams and Tri-grams
docs = docs_with_grams(df_clean['no_stop'].values)

# Create corpus and dictionary
dictionary, corpus = get_dict_corpus(docs)
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))
print(corpus[:1])
```

```ruby
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=9, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
# Print the Keyword in the 9 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```
![Topic](/images/topic.jpg)

After we have the base model, let fine tune the model then pick the number of topic we should use for model
Here, at the result of tunning model, I'll divine the data into 5 topic for analysing
(Please check the detail code of tunning part in my EDA.jynb)
Visualize the topic
```
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
LDAvis_prepared
```
![LDAvis](/images/LDAvis.jpg)

## 6. Nearest Neighbors

```ruby
# Document:
text_array = df_clean['no_stop']

# Topic vector in document:
recipe_vecs = []
for i in range(text_array.shape[0]):
    top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = list(map(lambda x:x[1], top_topics))
    recipe_vecs.append(topic_vec)
recipe_vecs[1]
```
![recipe_vecs](/images/recipe_vecs.jpg)

```ruby
# Nearest Neighbors:
nbrs = NearestNeighbors(n_neighbors=5).fit(recipe_vecs)
nbrs_path = '/content/gdrive/MyDrive/Colab Notebooks/Final Project/Model/nbrs_5.pkl'
pickle.dump(nbrs, open(nbrs_path, 'wb'))
```

```ruby
# Function to lemmatize for string input test doc
def lemma_stop_input(row):
    return ' '.join([lemmatizer.lemmatize(word) for word in row if word not in stop_words])

# Function to lemmatize & remove stop_words of test doc:
def preprocess_text(test_array):
    ''' Preprocess input text
        Output: list of vector of input text '''
    # Lemmatize and remove stop words    
    test_array = [lemma_stop(test_array)]
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
```
```ruby
# Input and preprocess test doc
test_array = text_array[5]

# Vector of test doc
result_vecs = doc_vecs(test_array)
result_vecs
```
![result_vecs](/images/result_vecs.jpg)

```ruby
# Recipe vector & their topics:
topic_recipe = pd.DataFrame(recipe_vecs)
topics = np.argmax(recipe_vecs,axis=1)
topic_recipe['topics'] = topics
topic_recipe
```
![topic_recipe](/images/topic_recipe.jpg)

```ruby
# Creat topic dictionary base on model
topic_dict = {0: 'fruit_juice',
              1: 'cocktail',
              2: 'liqueur',
              3: 'cream_milk_coffee',
              4: 'spice'}
```

```ruby
# Find the nearest vector 
distances, indices = nbrs.kneighbors(result_vecs)
print(distances)
print(indices)
```
![distance_indice](/images/distance_indice.jpg)

```ruby
# Recommend topics for the test doc
topic_recipe.iloc[indices[0]]
```
![recommend_topic](/images/recommend_topic.jpg)

```ruby
# Check top 5 recommended topics and recipes informations:
for i in indices[0]:
    print('Topic:' + topic_dict[topic_recipe['topics'][i]])
    print(df_clean.iloc[i]['drink_name'] + ': ' + df_clean.iloc[i]['recipe'])
```
![recipe_info](/images/recipe_info.jpg)
