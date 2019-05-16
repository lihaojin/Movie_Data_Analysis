from IPython.display import Image, HTML
import datasets
import json
import datetime
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
from ast import literal_eval
from scipy import stats
import altair as alt
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

def profitByGenre():
    """ Bar Graph showing the total profit per genre """
    meta_df = datasets.getMetadata()
    meta_df['profit_loss'] = meta_df['revenue'].sub(meta_df['budget'], axis = 0) 
    meta_df['profit_loss'].fillna(0.0, inplace = True)
    genre_dict = {}
    """ Create genres dictionary """
    for index, row in meta_df.iterrows():
        genre_length = len(literal_eval(meta_df.movie_genres[index]))
        for i in range(genre_length):
            if literal_eval(meta_df.movie_genres[index])[i] in genre_dict:
                genre_dict[literal_eval(meta_df.movie_genres[index])[i]].append(meta_df['profit_loss'][index])
            else:
                genre_dict[literal_eval(meta_df.movie_genres[index])[i]] = []
                genre_dict[literal_eval(meta_df.movie_genres[index])[i]].append(meta_df['profit_loss'][index])
    # caculate total profit for each genre
    for genre in genre_dict:
        genre_dict[genre] = np.sum(genre_dict[genre])

    genre, genre_profit = zip(*genre_dict.items())
    genre_df = pd.DataFrame({'genre':genre, 'profit':genre_profit})
    genre_df.genre = genre_df.genre.astype(str)
    """ total profit of each genre """
    return alt.Chart(genre_df).mark_bar().encode(
                x='genre',
                y='profit',
                color='genre'
            )

def genreAvgRevenueWordCloud():
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    revenueDict = {}
    revenueCount = {}
    def genreAvgRevenue(row):
        if(not pd.isna(row['revenue'])):
            for i in ast.literal_eval(row['genres']):
                if(i in revenueDict):
                    revenueDict[i] = revenueDict[i] + row['revenue']
                    revenueCount[i] = revenueCount[i] + 1
                else:
                    revenueDict[i] = row['revenue']
                    revenueCount[i] = 1
    movieRevenue = pd.DataFrame({'genres': df['genres'], 'revenue': df['revenue']})

    for index, row in movieRevenue.iterrows():
        genreAvgRevenue(row)
    avgRevenue = {}
    for i in revenueDict:
        avgRevenue[i] = revenueDict[i]/revenueCount[i]
    wordcloud = WordCloud(width=1050,height=900, background_color='white',
                          max_words=1628,relative_scaling=0.7,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(avgRevenue)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


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
    df['genres'].apply(genreCount)
    wordcloud = WordCloud(width=1050,height=900, background_color='white',
                      max_words=1628,relative_scaling=0.7,
                      normalize_plurals=False)
    wordcloud.generate_from_frequencies(genreDict)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def decadePiechart():

    def stats(gr):
        return {'min':gr.min(),'max':gr.max(),'count': gr.count(),'mean':gr.mean()}

    def label(s):
        val = (1900 + s, s)[s < 100]
        chaine = '' if s < 30 else "{}'s".format(int(val))
        return chaine

    df = pd.read_csv('./clean_datasets/clean_metadata.csv')
    df['decade'] = df['year'].apply(lambda x:((x-1900)//10)*10)

    df = df.drop(df[df.decade < 10].index)
    df = df.drop(df[df.decade > 110].index)

    test = df['year'].groupby(df['decade']).apply(stats).unstack()

    sns.set_context("poster", font_scale=0.85)

    plt.rc('font', weight='bold')
    f, ax = plt.subplots(figsize=(11, 6))
    labels = [label(s) for s in  test.index]
    sizes  = test['count'].values
    explode = [0.2 if sizes[i] < 100 else 0.01 for i in range(11)]
    ax.pie(sizes, explode = explode, labels=labels,
           autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
           shadow=False, startangle=0)
    ax.axis('equal')
    plt.show()
