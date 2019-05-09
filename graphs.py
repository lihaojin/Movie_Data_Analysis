%matplotlib inline
%run datasets.py
from IPython.display import Image, HTML
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

def createGraphs():
    df = getMetadata()
    
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
    
    #Setting months and days to create another set of graphs based on time.
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
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
    
    df['day'] = df['release_date'].apply(get_day)
    df['month'] = df['release_date'].apply(get_month)
    
    #Creating a bar graph depicting movies released per month.
    plt.figure(figsize=(12,6))
    plt.title("Number of Movies released in a particular month.")
    sns.countplot(x='month', data=df, order=month_order)
    
    #Another bar graph - this one showing revenue per month for blockbuster movies.
    month_mean = pd.DataFrame(df[df['revenue'] > 1e8].groupby('month')['revenue'].mean())
    month_mean['mon'] = month_mean.index
    plt.figure(figsize=(12,6))
    plt.title("Average Gross by the Month for Blockbuster Movies")
    sns.barplot(x='mon', y='revenue', data=month_mean, order=month_order)
