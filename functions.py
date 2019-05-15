import pandas as pd
import math, nltk, warnings
from nltk.corpus import wordnet
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
PS = nltk.stem.PorterStemmer()
gaussian_filter = lambda x,y,sigma: math.exp(-(x-y)**2/(2*sigma**2))

def count_word(df, ref_col, list):
    keyword_count = dict()
    for s in list: keyword_count[s] = 0
    for list_keywords in df[ref_col].str.split('|'):
        if type(list_keywords) == float and pd.isnull(list_keywords): continue
        #for s in list:
        for s in [s for s in list_keywords if s in list]:
            if pd.notnull(s): keyword_count[s] += 1
    #______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

def keywords_inventory(dataframe, column = 'keywords'):
    PS = nltk.stem.PorterStemmer()
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys = []
    icount = 0
    for s in dataframe[column]:
        if pd.isnull(s): continue
        for t in s.split('|'):
            t = t.lower() ; racine = PS.stem(t)
            if racine in keywords_roots:                
                keywords_roots[racine].add(t)
            else:
                keywords_roots[racine] = {t}
    
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)            
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    print("Nb of keywords in variable '{}': {}".format(column,len(category_keys)))
    return category_keys, keywords_roots, keywords_select

# Replacement of the keywords by the main form
#----------------------------------------------
def replacement_df_keywords(df, dico_replacement, roots = False):
    df_new = df.copy(deep = True)
    for index, row in df_new.iterrows():
        chain = row['keywords']
        if pd.isnull(chain): continue
        new_list = []
        for s in chain.split('|'): 
            clef = PS.stem(s) if roots else s
            if clef in dico_replacement.keys():
                new_list.append(dico_replacement[clef])
            else:
                new_list.append(s)       
        df_new.set_value(index, 'keywords', '|'.join(new_list)) 
    return df_new

# Get the synomyms of the word 'mot_cle'
#--------------------------------------------------------------
def get_synonyms(mot_cle):
    lemma = set()
    for ss in wordnet.synsets(mot_cle):
        for w in ss.lemma_names():
            #_______________________________
            # We just get the 'nouns':
            index = ss.name().find('.')+1
            if ss.name()[index] == 'n': lemma.add(w.lower().replace('_',' '))
    return lemma

# Checks if 'mot' is a key of 'key_count' with a test on the number of occurences   
#----------------------------------------------------------------------------------
def test_keyword(mot, key_count, threshold):
    return (False , True)[key_count.get(mot, 0) >= threshold]

def replacement_df_low_frequency_keywords(df, keyword_occurences):
    df_new = df.copy(deep = True)
    key_count = dict()
    for s in keyword_occurences: 
        key_count[s[0]] = s[1]    
    for index, row in df_new.iterrows():
        chain = row['keywords']
        if pd.isnull(chain): continue
        new_list = []
        for s in chain.split('|'): 
            if key_count.get(s, 4) > 3: new_list.append(s)
        df_new.set_value(index, 'keywords', '|'.join(new_list))
    return df_new

# Returns the values taken by the variables 'director_name', 'actor_N_name' (N ∈ [1:3]) and 'plot_keywords' for the film selected by the user.
def entry_variables(df, id_entry): 
    col_labels = []    
    if pd.notnull(df['directors'].iloc[id_entry]):
        for s in df['directors'].iloc[id_entry].split('|'):
            col_labels.append(s)
            
    for i in range(5):
        column = 'actorNUM'.replace('NUM', str(i+1))
        if pd.notnull(df[column].iloc[id_entry]):
            for s in df[column].iloc[id_entry].split('|'):
                col_labels.append(s)
                
    if pd.notnull(df['keywords'].iloc[id_entry]):
        for s in df['keywords'].iloc[id_entry].split('|'):
            col_labels.append(s)
    return col_labels

# Adds a list of variables to the dataframe given in input and initializes these variables at 0 or 1 depending on the correspondance with the description of the films and the content of the REF_VAR variable given in input.
def add_variables(df, REF_VAR):    
    for s in REF_VAR: df[s] = pd.Series([0 for _ in range(len(df))])
    columns = ['genres', 'actor1', 'actor2',
                'actor3', 'actor4', 'actor5', 'directors', 'keywords']
    for category in columns:
        for index, row in df.iterrows():
            if pd.isnull(row[category]): continue
            for s in row[category].split('|'):
                if s in REF_VAR: df.set_value(index, s, 1)            
    return df

# Creates a list of N(= 31) films similar to the film selected by the user.
def recommand(df, id_entry):    
    df_copy = df.copy(deep = True)    
    list_genres = set()
    for s in df['genres'].str.split('|').values:
        list_genres = list_genres.union(set(s))    
    #_____________________________________________________
    # Create additional variables to check the similarity
    variables = entry_variables(df_copy, id_entry)
    variables += list(list_genres)
    df_new = add_variables(df_copy, variables)
    #____________________________________________________________________________________
    # determination of the closest neighbors: the distance is calculated / new variables
    X = df_new.as_matrix(variables)
    nbrs = NearestNeighbors(n_neighbors=31, algorithm='auto', metric='euclidean').fit(X)

    distances, indices = nbrs.kneighbors(X)    
    xtest = df_new.iloc[id_entry].as_matrix(variables)
    xtest = xtest.reshape(1, -1)

    distances, indices = nbrs.kneighbors(xtest)

    return indices[0][:]

# Extracts some variables of the dataframe given in input and returns this list for a selection of N films. This list is ordered according to criteria established in the criteria_selection() function.
def extract_parameters(df, list_films):     
    parametres_films = ['_' for _ in range(31)]
    i = 0
    max_users = -1
    for index in list_films:
        parametres_films[i] = list(df.iloc[index][['title', 'year', 
                                                   'vote_average',
                                                   'original_language',
                                                   'vote_count']])
        parametres_films[i].append(index)
        max_users = max(max_users, parametres_films[i][4] )
        i += 1
        
    title_main = parametres_films[0][0]
    annee_ref  = parametres_films[0][1]
    parametres_films.sort(key = lambda x:criteria_selection(title_main, max_users,
                                    annee_ref, x[0], x[1], x[2], x[4]), reverse = True)

    return parametres_films 

# Compares the 2 titles passed in input and defines if these titles are similar or not.
def sequel(title_1, title_2):    
    if fuzz.ratio(title_1, title_2) > 50 or fuzz.token_set_ratio(title_1, title_2) > 50:
        return True
    else:
        return False

# Gives a mark to a film depending on its IMDB score, the title year and the number of users who have voted for this film.
def criteria_selection(title_main, max_users, annee_ref, title, annee, imdb_score, votes):    
    if pd.notnull(annee_ref):
        factor_1 = gaussian_filter(annee_ref, annee, 20)
    else:
        factor_1 = 1        

    sigma = max_users * 1.0

    if pd.notnull(votes):
        factor_2 = gaussian_filter(votes, max_users, sigma)
    else:
        factor_2 = 0
        
    if sequel(title_main, title):
        note = 0
    else:
        note = imdb_score**2 * factor_1 * factor_2
    
    return note

# Complete the film_selection list which contains 5 films that will be recommended to the user. The films are selected from the parametres_films list and are taken into account only if the title is different enough from other film titles.
def add_to_selection(film_selection, parametres_films):    
    film_list = film_selection[:]
    icount = len(film_list)    
    for i in range(31):
        already_in_list = False
        for s in film_selection:
            if s[0] == parametres_films[i][0]: already_in_list = True
            if sequel(parametres_films[i][0], s[0]): already_in_list = True            
        if already_in_list: continue
        icount += 1
        if icount <= 3:
            film_list.append(parametres_films[i])
    return film_list

# Removes sequels from the list if more than two films from a series are present. The older one is kept.
def remove_sequels(film_selection):    
    removed_from_selection = []
    for i, film_1 in enumerate(film_selection):
        for j, film_2 in enumerate(film_selection):
            if j <= i: continue 
            if sequel(film_1[0], film_2[0]): 
                last_film = film_2[0] if film_1[1] < film_2[1] else film_1[0]
                removed_from_selection.append(last_film)

    film_list = [film for film in film_selection if film[0] not in removed_from_selection]

    return film_list   

# Creates a list of 5 films that will be recommended to the user.
def find_similarities(df, id_entry, del_sequels = True, verbose = False):    
    if verbose: 
        print(90*'_' + '\n' + "QUERY: films similar to id={} -> '{}'".format(id_entry,
                                df.iloc[id_entry]['title']))
    #____________________________________
    list_films = recommand(df, id_entry)
    #__________________________________
    # Create a list of 31 films
    parametres_films = extract_parameters(df, list_films)
    #_______________________________________
    # Select 5 films from this list
    film_selection = []
    film_selection = add_to_selection(film_selection, parametres_films)
    #__________________________________
    # delation of the sequels
    if del_sequels: film_selection = remove_sequels(film_selection)
    #______________________________________________
    # add new films to complete the list
    film_selection = add_to_selection(film_selection, parametres_films)
    #_____________________________________________
    selection_titles = []
    for i,s in enumerate(film_selection):
        selection_titles.append([s[0].replace(u'\xa0', u''), s[3]])
        if verbose: print("nº{:<2}     -> {:<30}".format(i+1, s[0]))

    return selection_titles
