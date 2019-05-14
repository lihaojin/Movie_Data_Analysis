from IPython.display import Image, HTML
import datasets
import json
import datetime
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
sns.set_style('whitegrid')
sns.set(font_scale=1.25)
pd.set_option('display.max_colwidth', 50)

df = datasets.getMetadata()
ratings_df = datasets.getRatings()

#Setting months and days to create another set of graphs based on time.
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
# 
# df['day'] = df['release_date'].apply(get_day)
# df['month'] = df['release_date'].apply(get_month)
#

def byLanguage():
    #Checking for counts of movies in each language.
    lang_df = pd.DataFrame(df['original_language'].value_counts())
    lang_df['language'] = lang_df.index
    lang_df.columns = ['number', 'language']
    lang_df.head()

    #Creates a bar graph visualizing the number of movies per language.
    plt.figure(figsize=(12,5))
    sns.barplot(x='language', y='number', data=lang_df.iloc[0:11])
    plt.title('Number of movies per language')
    plt.show()

def get_month(x):
    try:
        return month_order[int(str(x).split('-')[1]) - 1]
    except:
        return np.nan

def get_day(x):
    try:
        year, month, day = (int(i) for i in x.split('-'))
        answer = datetime.date(year, month, day).weekday()
        return day_order[answer]
    except:
        return np.nan

def byMonth():
    #Creating a bar graph depicting movies released per month.
    plt.figure(figsize=(12,6))
    plt.title("Number of Movies released in a particular month.")
    sns.countplot(x='month', data=df, order=month_order)

def byRevenue():
    #Another bar graph - this one showing revenue per month for blockbuster movies.
    month_mean = pd.DataFrame(df[df['revenue'] > 1e8].groupby('month')['revenue'].mean())
    month_mean['mon'] = month_mean.index
    plt.figure(figsize=(12,6))
    plt.title("Average Gross by the Month for Blockbuster Movies")
    sns.barplot(x='mon', y='revenue', data=month_mean, order=month_order)

def profit_by_genre():
    df = df.rename(index=str, columns={"id": "movieId"})
    df.movieId = df.movieId.astype(int)
    df.budget = df.budget.astype(float)
    df.popularity = df.popularity.astype(float)
    movies_ratings_df = pd.merge(df, ratings_df, on='movieId')

    # calculate the profit/loss for each movie
    movies_ratings_df['profit_loss'] = movies_ratings_df['revenue'].sub(movies_ratings_df['budget'], axis = 0)

    # replace NaN with 0
    movies_ratings_df['profit_loss'] = movies_ratings_df['profit_loss'].fillna(0.0)

    # create a dictionary of each genre & retrieve the profit/loss for each genre
    genre_dict = {}
    for row in range(len(movies_ratings_df)):
        genre_length = len(movies_ratings_df.movie_genres[row])
        for i in range(genre_length):
            if movies_ratings_df.movie_genres[row][i] in genre_dict:
                genre_dict[movies_ratings_df.movie_genres[row][i]].append(movies_ratings_df['profit_loss'][row])
            else:
                genre_dict[movies_ratings_df.movie_genres[row][i]] = []
                genre_dict[movies_ratings_df.movie_genres[row][i]].append(movies_ratings_df['profit_loss'][row])

    # caculate total profit for each genre
    for genre in genre_dict:
        genre_dict[genre] = np.sum(genre_dict[genre])

    # drop brackets that were stored as a genre
    genre_dict.pop('[')
    genre_dict.pop(']')
    
    genre, genre_profit = zip(*genre_dict.items())
    genre_df = pd.DataFrame({'genre':genre, 'profit':genre_profit})
    genre_df.genre = genre_df.genre.astype(str)
    
    """ total profit of each genre """
    alt.Chart(genre_df).mark_bar().encode(
        x='genre',
        y='profit',
        color='genre'
    )


def genreWordCloud():
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    genreDict = {}
    def genreCount(genres):
        for i in ast.literal_eval(genres):
            if(i in genreDict):
                genreDict[i] = genreDict[i] + 1
            else:
                genreDict[i] = 1
    df['movie_genres'].apply(genreCount)
    wordcloud = WordCloud(width=1050,height=900, background_color='white',
                      max_words=1628,relative_scaling=0.7,
                      normalize_plurals=False)
    wordcloud.generate_from_frequencies(genreDict)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
    