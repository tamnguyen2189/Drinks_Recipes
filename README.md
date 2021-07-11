# Your_Recipe
## Introduction
Baristas and bartenders have to develop new recipes to refresh their menu constantly. Therefore, this project will help them solve that problem. In addition, it will bring more ideas and inspiration for new and innovative drinks. 

Your Recipe is an app where you can input ingredients, and it will return delicious recipes based on your selections. The app also suggests other similar drinks to expand upon your search results.
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

For this project, I will build a Doc2Vec model, and an LDA topic model using a dataset scraped from The Spruce Eats and All Recipes. This dataset contains various drink ingredients, such as fruit, juice, milk, coffee, etc. We can use these models to find the closest matching recipes from the dataset and recommend similar items. 
![Dataset](/images/dataset.jpg)

## 2. Reading and Cleaning Data
**Reading Data**

Let's take a look at the content of the dataset. I saved recipes into many files csv, so first, we need to concatenate the files.

```ruby
import pandas as pd
import numpy as np
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

**Cleaning Data**

Since I scraped all the information needed, we don't need to drop any columns. Therefore, we'll focus on the text data, including the *"drink_name"* column and *"recipe"* column. But, there are still some recipes that contain incorrect data, duplicated data, and NA data. So we need to drop all of them.
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
**Punctuation and Lowercasing**

Next, let's perform a simple preprocessing on the content of the paper_text column to make them more manageable for analysis and reliable results. To do that, we'll use a regular expression to remove any punctuation and then lowercase the text. We also need to drop some recipes that have any error URLs for the image.
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

**Saving Data**

Remember to save the dataset after cleaning. So from now on, we can use this clean data to train the model.
```ruby
# Save data for training after cleaning
data = df_clean
path = '/content/gdrive/MyDrive/Colab Notebooks/Final Project/Recipes'
data.to_csv(path + "/clean_recipe_training.csv", index=False)
```

## 3. Searching Recipe by Pandas
At this step, I use pandas to find all recipes that contain the exact ingredients for each user input. But, first, let us define the function for searching:
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
The first model I used inside Your Recipe is the Doc2Vec model, a modified version of Word2Vec and is used to represent documents in numeric values. 

There are multiple methods to change the text into vectors:
- Label Encoding
- Custom binary Encoding
- One-Hot Encoding

Except, these methods will lose the context of a given text. Doc2Vec can solve this problem by creating vectors out of a document, independent of the document length. 
![Doc2Vec](/images/doc2vec.jpg)

Distributed Memory (DM) model will guess the target word from its neighboring words (context words). But the document doesn't have a logical structure like a word. So to solve this problem, it adds another vector called the document ID.

Distributed Bag of Words (DBOW) model, which guesses the context words from a target word.

Doc2vec model takes the document ID as the input and tries to predict randomly sampled words from the document. Then, we can use *most_similar* function to find most similar Recipe from dataset base on input doc.

```ruby
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
```

**Preprocessing data**

*Getting training data*

We define our list of sentences. Here is the column of recipes. It's good to use a list of sentences for easier processing of each sentence. 
We will also keep a tokenized version of these sentences.

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

Let's initialize the model and train it:
```ruby
# Train doc2vec model
model = Doc2Vec(tagged_data, vector_size = 50, window = 2, min_count = 2, epochs = 30)
```
Now, we give a test sentence. The infer_vector function returns the vectorized form of test sentence.

In the end, let's call the most_similar function, which returns topmost similar sentences and their indices throughout the document.

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
It divides the documents into several clusters according to word usage to find the topics in these documents.

LDA model also present the probabilistic distribution of Topics in Document.

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
**Remove Stopwords & Make Lemmatize**

Let’s define the functions to remove the stopwords, make lemmatization and call them sequentially. I also creat a new column, *'no_stop'* that contains recipes after cleaning stop words
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

Bigrams are two words frequently occurring together in the document; Trigrams are three words frequently occurring. Gensim Phrases model can build and implement the bigrams, trigrams, quadgrams, and more. The two essential arguments to Phrases are min_count and threshold.

The higher the values of these param, the harder it is for words to be combined. Let's define the function for Bigram and Trigram models.

_*Corpus and Dictionary*_

The two main inputs to the LDA topic model are the dictionary(id2word) and the corpus. Let’s create them:
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

Gensim creates a unique id for each word in the document. Thus, the produced corpus shown above is a mapping of (word_id, word_frequency).

**Base Model**

We have everything required to train the base LDA model. In addition to the corpus and dictionary, you need to provide the number of topics as well. First, I'll choose 9 topics:
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
**View Topics in LDA model**

The above LDA model is built with 10 different topics where each topic is a combination of keywords and each keyword contributes a certain weightage to the topic.
You can see the keywords for each topic and the weightage(importance) of each keyword using lda_model.print_topics()
```ruby
# Print the Keyword in the 9 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```
![Topic](/images/topic.jpg)

After we have the base model, let's fine-tune the model then pick the topic we should use.
Finally, as a tuning model, I'll define the data into five topics for analysis.

Visualize the topic:
```
# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
LDAvis_prepared
```
![LDAvis](/images/LDAvis.jpg)

## 6. Nearest Neighbors

Here we're going to convert each recipe in dataset to feature vectors:

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

Then, we use Nearest Neighbors to find the Top 5 recipes similar to the recipe input for each topic. The k here is the number of nearest-neighbor you want because I want results are Top 5 nearest recipes, so I'll choose k = 5

```ruby
# Nearest Neighbors:
nbrs = NearestNeighbors(n_neighbors=5).fit(recipe_vecs)
nbrs_path = '/content/gdrive/MyDrive/Colab Notebooks/Final Project/Model/nbrs_5.pkl'
pickle.dump(nbrs, open(nbrs_path, 'wb'))
```
Let's define the function to preprocess the input recipe and convert it to vector.
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
At this step, we will take a random recipe as a test document, then preprocess it.
```ruby
# Input and preprocess test doc
test_array = text_array[5]

# Vector of test doc
result_vecs = doc_vecs(test_array)
result_vecs
```
![result_vecs](/images/result_vecs.jpg)

We need to create a topics dictionary so we can categorize the topics. I'll base on the word frequency of each topic to give a name for these topics in the dataset.
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
We need to use the Nearest Neighbors model and its cosine similarity function to calculate distances and indices of those similar recipes base on the test document vector.
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

Finally, let's see which topic the test document belongs to and the Top 5 similar recipes.
```ruby
# Check top 5 recommended topics and recipes informations:
for i in indices[0]:
    print('Topic:' + topic_dict[topic_recipe['topics'][i]])
    print(df_clean.iloc[i]['drink_name'] + ': ' + df_clean.iloc[i]['recipe'])
```
![recipe_info](/images/recipe_info.jpg)
