#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import ast


# In[4]:


def movieData():
    #Read csv file
    df = pd.read_csv('./data/movies_metadata.csv')
    
    #Drop columns
    cols_to_drop = ['homepage', 'tagline', 'poster_path']
    df = df[df.columns.drop(cols_to_drop)]
    
    #Drop rows with EOF error
    df = df.drop([19729, 19730, 29502, 29503, 35586, 35587])
    
    #Function to generate column with only 'name' key
    def getLang (langs):  
        x = ast.literal_eval(langs)
        if(len(x) != 0):
            new_x = []
            for i in range(len(x)):
                name = list(x[i].get("name"))
                new_x.append(x[i].get("name"))

            return(new_x)

        return langs
    
    #Make languages column
    languages = list(df.spoken_languages)
    lang_df = []

    for i in range(0, len(languages), 1):
        try:
            #lang = []
            lang_df.append(getLang(languages[i]))
            #lang_df.append(lang)

        except:
            print(i)
    
    #Add languages column to dataframe
    df = df.assign(languages = lang_df)

    #Drop original languages column
    df = df.drop(columns = ['spoken_languages'])
    
    
    #Make genres column
    genres = list(df.genres)
    genres_df = []

    for i in range(0, len(genres), 1):
        try:
            genres_df.append(getLang(genres[i]))

        except:
            print(i)
    
    #Add genres column to dataframe
    df = df.assign(movie_genres = genres_df)
    
    #Drop original genres column
    df = df.drop(columns = ['genres'])

    
    countries = list(df.production_countries)
    countries_df = []

    #Make countries column 
    for i in range(0, len(countries), 1):
        try:
            countries_df.append(getLang(countries[i]))

        except:
            print(i)
            
    #Add countries column to dataframe
    df = df.assign(countries = countries_df)
    
    #Drop original countries column
    df = df.drop(columns = ['production_countries'])

    
    #Make companies column
    companies = list(df.production_companies)
    companies_df = []

    for i in range(0, len(companies), 1):
        try:
            companies_df.append(getLang(companies[i]))

        except:
            print(i)
    
    #Add companies column
    df = df.assign(companies = companies_df)
    
    #Drop original companies column
    df = df.drop(columns = ['production_companies'])

    
    return df

