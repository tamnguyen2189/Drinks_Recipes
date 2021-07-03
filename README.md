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

