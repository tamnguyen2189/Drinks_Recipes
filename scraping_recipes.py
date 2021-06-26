# Import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Create function to scrape 1st web
url = 'https://www.thespruceeats.com/'
# Get image of drink
def scrape_image(url):
    """ Scrape image of the drink in recipe
        Input: recipe url
        Output: image url
    """
    soup = get_url(url)
    article = soup.find('div', {'class':'img-placeholder'})
    image_url = article.img['src']
    return image_url

# Get recipe of drink
def scrape_all_recipe(url):
    """ Scrape all ingredients of recipe in both way
        Input: url
        Output: list of ingredients
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    info = soup.find('section',{'class': 'comp section--fixed-width section--ingredients section'})
    
    ingredient = []
    try:
        content_part = info.find_all('li',{'class': 'structured-ingredients__list-item'})
        content_sentence = info.find_all('li',{'class': 'simple-list__item js-checkbox-trigger ingredient text-passage'})
        # for i in content:
        # if len(i.find_all('span')) != 0:
        if content_part != []:
            for i in content_part:
                item = i.find_all('span')
                sentence = []
                for j in item:
                    text = j.text
                    sentence.append(text)
                sentence = ' '.join(sentence)
                ingredient.append(sentence)
        else:
            for x in content_sentence:
                text = x.text
                text = text.strip('\n')
                ingredient.append(text) 
    except Exception as e:
        print(e)

    return ingredient

# Functions to extract recipes (sub/each category):
# Get recipes from sub article
def extract_sub_recipe(url):
    """ Extract info from all recipes of each link from website
        Input: sub article url
        Output: info of recipes, saved as list of dictionary.
    """
    
    data = []
    
    soup = get_url(url)
    content = soup.find('ul', {'class':'comp ordered-list--base ordered-list__list no-arrows structured-content article-content ordered-list--structured mntl-block'})
    info = content.find_all('a', {'class': 'mntl-sc-block-heading__link'})

    for i in info:
        d = {}
        try:
            d['drink_name'] = i.text
            d['recipe'] = scrape_all_recipe(i['href'])
            d['recipe_url'] = i['href']
            d['url_of_image'] = scrape_image(i['href'])
        except Exception as e:
            print(e)
        data.append(d)

    return data

# Get recipes from main category
def extract_category_recipe(url):
    """ Extract info from all recipes of each category
        Input: url - use scrape_recipe_parts()
        Output: info of recipes, saved as list of dictionary.
    """
    
    data = []
    
    soup = get_url(url)
    info = soup.find_all('li', {'class': 'comp masonry-list__item mntl-block'})

    for i in info:
        d = {}
        try:
            d['drink_name'] = i.span.text.strip('\n\n')
            d['recipe'] = scrape_all_recipe(i.a['href'])
            d['recipe_url'] = i.a['href']
            d['url_of_image'] = scrape_image(i.a['href'])
        
        except:
            print(i.a['href'])
        data.append(d)

    return data

# Create function to loop and scrape data from article list: 
def extract_whole_recipes(url):
    """ Extract info from all recipes of each category
        Input: url
        Output: info of recipes, saved as list of dictionary.
    """
    
    data = []
    
    soup = get_url(url)
    info = soup.find_all('li', {'class': 'comp masonry-list__item mntl-block'})

    for i in info:
        d = {}
        sub_data = []
        try:

            if len(scrape_all_recipe(i.a['href'])) != 0:
                d['drink_name'] = i.span.text.strip('\n\n')
                d['recipe'] = scrape_all_recipe(i.a['href'])
                d['recipe_url'] = i.a['href']
                d['url_of_image'] = scrape_image(i.a['href'])
            else: 
                sub_data = extract_category_recipe(i.a['href']) 

        except Exception as e:
            print(i.a['href'])
        data.append(d)
        data = data + sub_data

    return data

# Creat function to scrape whole the website
def extract_recipes_all_category(url):
    """ Extract info from all recipes of website
        Input: web url
        Output: info of recipes, saved as list of dictionary.
    """
    soup = get_url(url)
    content = soup.find_all('ul',{'class': 'fullscreen-nav__sub-list'})
    category = content[1].find_all('li', {'class': 'fullscreen-nav__sub-list-item'})

    data = []
    for link in category:
        sub_data = extract_whole_recipes(link.a['href'])
        data.append(sub_data)

    return data

#########

# Create function to scrape 2nd web
url = 'https://www.allrecipes.com/recipes/134/drinks/coffee/'

# Scraping only recipe
def scrape_recipe(url):
    """ Scrape the ingredient of normal recipe
        Input: recipe url
        Output: list of ingredients
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    # articles = soup.find('section', {'class':'comp section--fixed-width section--ingredients section'})
    content = soup.find_all('li',{'class': 'ingredients-item'})
    ingredient = []
    for i in content:
        text = i.span.text
        text = text.strip('\n')
        ingredient.append(text)
    return ingredient

# Scraping recipes in category
def extract_category_recipe(url):
    """ Extract info from all recipes of each category
        Input: url - use scrape_recipe_parts()
        Output: info of recipes, saved as list of dictionary.
    """
    
    data = []
    
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    info = soup.find_all('div', {'class': 'component card card__category'})
    data = []
    for i in info:
        d = {}
        try:
            d['drink_name'] = i.a['title'] 
            d['recipe'] = scrape_recipe(i.a['href'])
            d['recipe_url'] = i.a['href']
            d['url_of_image'] = i.a.div['data-src']
        
        except Exception as e:
            print(i.a['href'])
        data.append(d)

    return data

# Creat function to scrape whole the website
def extract_recipes_all_category(url):
    """ Extract info from all recipes of website
        Input: web url
        Output: info of recipes, saved as list of dictionary.
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    content = soup.find_all('li',{'class': 'carouselNav__listItem recipeCarousel__listItem'})
    
    data = []
    for link in content:
        sub_data = extract_category_recipe(link.a['href'])
        data.append(sub_data)

    return data

# Save data
data = []
path = 'C:/Users/pc/Desktop/my_git/final_project/recipes'
items = pd.DataFrame(data = data, columns = data[0].keys())
items.to_csv(path + "/recipe_name.csv", index=False)