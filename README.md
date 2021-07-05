# Your_Recipe
## Introduction
Barista and bartender have to constantly develop new recipes to refresh their menu. Therefore, this project will help them solve that problem. It will bring more ideas and inspiration for your new drink.

Your Recipe is an app, where you can input your ingredients and it will return delicious recipes base on the ingredients you have. The app also suggest other drinks similar to the one you selected. You can get many results at the same time.
![Project Flow](/images/viber_image_2021-07-03_11-43-59.jpg)
# Project Implementation
1. Dataset
2. Reading and Cleaning Data
3. Searching recipe by pandas
4. Doc2Vec model
5. Latent Dirichlet Allocation model (LDA)
6. Nearest Neighbors

Please check the complete code in my notebook EDA.ipynb on GitHub
*****
## 1. Dataset

In this project, I will build a Doc2Vec model and a LDA topic model from a dataset that I scraped from The Spruce Eats and All Recipes. 
Then let use these model to find the most similar recipes to one we might like to use from the dataset, in order to recommend other things we could use.
This dataset contains variety of topcis in drinks ingredients, like fruit, juice, milk, coffee...
![Dataset](/images/dataset.jpg)

## 2. Reading and cleaning Data
**Reading data**

Let's take a look at the content of the dataset. I saved recipes into many files csv, so we need to concatenate them first.

```ruby
import pandas as pd
import numpy as np
import glob
import re

# Reading data: concatenate all recipes into one file
path = r'/content/gdrive/MyDrive/Colab Notebooks/Final Project/Recipes' # use your path
files = glob.glob(path + "/*.csv")
all_files = []

for filename in files:
     df_sub = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8')
     all_files.append(df_sub)

frame = pd.concat(all_files, axis=0, ignore_index=True)
df = frame.copy()
```

**Cleaning data**

Since I've scraped all informations I need, so we don't need to drop any columns. And we'll focus on the text data which are "drink_name" column and "recipe" column.
But there are still some recipes that contain wrong data, duplicated data and NA data. So we need to drop all of them.
```ruby
# Drop wrong data
df.drop(df[df['recipe'] == '[]'].index, inplace= True)

# Drop duplicated data
df.drop_duplicates(subset ='recipe',
                     keep = 'first', inplace = True)
                     
# Drop all na values
df.dropna(axis=0, inplace=True)

df_clean = df.copy() # Make a copy of data
```
**Remove punctuation/lower casing**

Next, let’s perform a simple preprocessing on the content of paper_text column to make them more amenable for analysis, and reliable results. To do that, we’ll use a regular expression to remove any punctuation, and then lowercase the text. We also need to drop some recipes that have error url of image
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

**Saving data**

Remember to save the dataset after clean. So from now on, we can use this clean data to train the model.
```ruby
# Save data for training after cleaning
data = df_clean
path = '/content/gdrive/MyDrive/Colab Notebooks/Final Project/Recipes'
data.to_csv(path + "/clean_recipe_training.csv", index=False)
```

## 3. Searching recipe by pandas
At this part, I use pandas to find all recipes that contain exactly each ingredient user input. Let define the function for searching:
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
The first model I use inside Your Recipe is Doc2Vec model. It’s a modified version of Word2Vec and is used to represented document into numeric value.

There are multiple methods to change the text into vectors:
- Label Encoding
- Custom binary Encoding
- One-Hot Encoding

But these methods will lose the context of a given text. Doc2Vec can solve this problem by creating vectors out of a document, independent of the document length.
![Doc2Vec](/images/doc2vec.jpg)

Distributed Memory (DM) model will guess the target word from its neighboring words (context words). But document doesn’t have logical structure like word. To solve this problem, it add another vector called document ID
Distributed Bag of Words (DBOW) model which guesses the context words from a target word 

Doc2vec model takes the document ID  as the input and tries to predict randomly sampled words from the document. Then, we can use *most_similar* function to find most similar recipe from dataset base on input doc

```ruby
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
```

**Preprocessing data**

*Getting training data*

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

# Get the tagged document
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
```
![Tagged_data](/images/tag_doc.jpg)

*Training model*

Let's initialize the model and train it
```ruby
# Train doc2vec model
model = Doc2Vec(tagged_data, vector_size = 50, window = 2, min_count = 2, epochs = 30)
```
Now, we give a test sentence. The infer_vector function returns the vectorized form of test sentence.

In the end, let's call the most_similar function, which returns top most similar sentences and its indices throughout the document.

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
![LDA](/images/LDA.jpg)
The second model I use here is LDA model.
It will divide the documents in a number of clusters according to word usage, to find the topics in these document.

LDA model also present the probabilistic distribution of Topics in Document

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
**Remove Stopwords, Make Lemmatize**

Let’s define the functions to remove the stopwords, make lemmatization and call them sequentially.
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
**Phrase Modeling & Data Transformation**

_*Bigram and Trigram Models*_

Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.
Gensim’s Phrases model can build and implement the bigrams, trigrams, quadgrams and more. The two important arguments to Phrases are min_count and threshold.

The higher the values of these param, the harder it is for words to be combined. Let's define the function for Bigram and Trigram models

_*Corpus and Dictionary*_

The two main inputs to the LDA topic model are the dictionary(id2word) and the corpus. Let’s create them.
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
Let’s call the functions in order.

- Corpus and Dictionary

- Bi-grams and Tri-grams
```ruby
# Phrase modeling: Bi-grams and Tri-grams
docs = docs_with_grams(df_clean['no_stop'].values)

# Create corpus and dictionary
dictionary, corpus = get_dict_corpus(docs)
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))
print(corpus[:1])
```
![Data_transform](/images/data_transform.jpg)

Gensim creates a unique id for each word in the document. The produced corpus shown above is a mapping of (word_id, word_frequency).

**Base Model**

We have everything required to train the base LDA model. In addition to the corpus and dictionary, you need to provide the number of topics as well. First, I'll choose 9 topics.
```ruby
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=9, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
```
**View the topics in LDA model**

The above LDA model is built with 10 different topics where each topic is a combination of keywords and each keyword contributes a certain weightage to the topic.
You can see the keywords for each topic and the weightage(importance) of each keyword using lda_model.print_topics()
```ruby
# Print the Keyword in the 9 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```
![Topic](/images/topic.jpg)

After we have the base model, let fine tune the model then pick the number of topic we should use for model
At the result of tunning model, I'll divine the data into 5 topics for analysing

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
