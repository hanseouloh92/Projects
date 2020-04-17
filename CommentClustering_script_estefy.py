# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:03:31 2019

@author: mhan1
"""

import re
import os
import yaml
import pandas as pd
import datetime as dt
import cx_Oracle
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from functools import reduce

# setting the working directory to read config.yaml file
os.chdir(r'C:\Users\mhan1\OneDrive - DFW Airport\Desktop\Projects\CommentCluster')
os.getcwd()

# parse through yaml config file
with open('./config.yml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc: # will throw an error if an issue occured on parse
        print(exc)
        
# setting up connection to Oracle DB Prod
conn = cx_Oracle.connect(config['oracle_prod']['usr'], config['oracle_prod']['pwd'], config['oracle_prod']['host']+":"+config['oracle_prod']['port']+"/"+config['oracle_prod']['service'])
cur = conn.cursor()

# query based on the mstr report
survey = pd.read_sql_query(""" PUT YOUR QUERY HERE """,conn)
    
# simple function that will iterate through the differen regular expressions and return a cleaned version of text
def clean_text(text):
    stopwords_ref = stopwords.words('english')
    stopwords_ref.extend(['need', 'near', 'like', 'around', 'area', 'would', 'areas', 'connection',
                          'get','needed', 'better', 'add', 'one', 'go', 'know',
                          'take', 'make', 'need', 'needs', 'clean', 'cleans', 'cleaned', 'cleaner',
                          'working', 'works', 'worked', 'work', 'use', 'difficult'])
    regex_list = ["TERMINAL ['A-E']", # removing terminal a/b/c/d/e
              "[A-E]\d+", # removing gatecodes
              "\d+:\d+AM", # removing time
              "\d+:\d+PM", # removing time
              "\d+:\d+ AM",
              "\d+:\d+ PM",
              "\d+", # remove left over single digits
              "TERMINAL",
              "TERMINALS",
              "GATE",
              "\\b[A-E]\\b"] # removing single instances of A/B/C/D/E 
              #"[^\w\s]"] # removes punctuation 
    new_text = text
    for regex in regex_list:
        new_text = re.sub(r'{}'.format(regex), '',new_text)
    new_text = ' '.join([word for word in new_text.lower().split() if word not in stopwords_ref])
    new_text = re.sub("[^\w\s]", '', new_text) # removes puncuation after removing stop words b/c stop words had conjunctions(don't, won't, etc)
    # replacing some synonyms with one word to group by
    new_text = new_text.lower().replace('signage', 'sign').replace('washroom', 'restroom').replace('bathroom', 'restroom')
    return new_text

# creating a tokenizer/stemmer which removes tense and splits words into individual parts
def tokenizer_stem(text):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in text.split()]

#bad_reviews = survey.loc[survey['MASTER_ANSWER_ID'] <= 2]
# extracting only the text columns
all_comments = survey['SURVERY_ANSWER_TXT_UPPER']

# applying the clean_text function to each comment in the dataframe
cleaned_text = all_comments.apply(clean_text)

# setting up a TFIDF and Count vectorizer which counts the number of words given the number of documents
tfidf_vect = TfidfVectorizer(max_df = 0.8, 
                             min_df = 5,
                             use_idf = True,
                             tokenizer = tokenizer_stem,
                             ngram_range = (1, 2))
cnt_vect = CountVectorizer(tokenizer = tokenizer_stem, ngram_range = (1,2))

# fitting the vectorizer to our dataset
bag_of_words = cnt_vect.fit_transform(cleaned_text)
tfidf = tfidf_vect.fit_transform(cleaned_text)

# setting the sparse array to a pandas dataframe
bow_df = pd.DataFrame(data = bag_of_words.toarray(), columns = cnt_vect.get_feature_names())
tfidf_df = pd.DataFrame(data = tfidf.toarray(), columns = tfidf_vect.get_feature_names())

# # sort values by the counts to see most/least common words in our dataset
bow = bow_df.sum().sort_values(ascending = True).reset_index().rename(columns = {0: 'BOW_CNT'})
tfidf = tfidf_df.sum().sort_values(ascending = False).reset_index().rename(columns = {0: 'TFIDF_CNT'})

merged = bow.merge(tfidf, on = 'index')

# ##### ELBOW CURVE START
# import matplotlib.pyplot as plt
# # creating an elbow curve to determine the most optimal amount of clusters to use for our dataset
# distortions = []
# K = range(1,50) # cluster chosen from 1 - 50

# # fit 20 kmean models and determine the amonut of variance explained
# for k in K:
#     print('Currently fitting ', k, 'clusters')
#     mdl = KMeans(n_clusters = k)
#     mdl.fit(tfidf_df)
#     distortions.append(mdl.inertia_)
    
# # plotting the data
# fig = plt.figure(figsize = (15,5))
# plt.plot(range(1,50), distortions)
# plt.grid(True)
# plt.title('Elbow Curve')
# ###### ELBOW CURVE END

num_clusters = 11
km_mdl = KMeans(n_clusters = num_clusters, random_state = 42)
km_mdl.fit(tfidf_df)
cluster = km_mdl.labels_.tolist()

# identify the centers of each cluster and sorting them smallest to largest
order_centroid = km_mdl.cluster_centers_.argsort()[:,::-1]

cluster_ref = []
print('Top Words per Cluster')
print('=====================')
for i in range(num_clusters):
    cluster_dict = {}
    print('Cluster %d words: \n' % (i), end = '' )
    cluster_dict['cluster'] = i
    words = []
    for j in order_centroid[i, :3]:
        print(' %s' % tfidf_vect.get_feature_names()[j])
        words.append(tfidf_vect.get_feature_names()[j])
        word_string = ';'.join(words)
    cluster_dict['words'] = word_string
    cluster_ref.append(cluster_dict)

# creating dictionary for the results from clustering algorithm
comments = {'terminal': survey['TERMINAL_SHORT_DESC'].tolist(),
            'gate': survey['LOCATION_GATE_DSC'].tolist(), 
            'passenger_type': survey['PASSENGER_TYPE_DESC'].tolist(), 
            'date': survey['TRANS_DATE_TIME_NO_SECS'].tolist(),
            'intl_dom': survey['INTL_DOM_SHORT_DESC'].tolist(), 
            'comment': survey['SURVERY_ANSWER_TXT_UPPER'].tolist(),
            'cluster': num_clusters}

# convert dictionary to dataframe
results = pd.DataFrame(comments, index = [cluster], columns =  ['terminal', 'gate', 'passenger_type', 'intl_dom', 'comment', 'date'])\
    .reset_index().rename(columns = {'level_0': 'clusters'})
results['cluster_words'] = results['clusters'].apply(lambda x: cluster_ref[x]['words'])


today = dt.datetime.now().date()
this_month = str(today.year) + '-' + str(today.month)
# filter past month comments
cur_month = results.loc[results['date'] >= this_month]
# group by and get the count of comments that fall within each cluster
output = cur_month.groupby(['terminal', 'passenger_type', 'intl_dom', 'clusters', 'cluster_words'])['comment'].count().reset_index()

#################################################################################################################### temporary thing for simon
output = output.groupby(['terminal', 'clusters', 'cluster_words'])['comment'].sum().reset_index()
total_comments_terminal = output.groupby(['terminal'])['comment'].sum().reset_index()

output = output.merge(total_comments_terminal, how = 'left', on = 'terminal')

output['percentage'] = output['comment_x'] / output['comment_y']
output = output.drop('comment_y', axis = 1).rename(columns = {'comment_x': 'comment'})

unique_clusters = list(output['clusters'].unique())
for terminal in output['terminal'].unique():
    terminal_clusters = list(output.loc[output['terminal'] == terminal]['clusters'].unique())
    miss_groups = pd.DataFrame(list(set(unique_clusters) - set(terminal_clusters))).rename(columns = {0: 'clusters'})
    if len(miss_groups) == 0:
        pass
    else:
        miss_groups['cluster_words'] = miss_groups['clusters'].apply(lambda x: cluster_ref[x]['words'])
        miss_groups['terminal'] = terminal
        miss_groups['percentage'] = 0
        miss_groups['comment'] = 0
        output = pd.concat([miss_groups, output])
    
output['rank'] = output.groupby('terminal')['percentage'].rank('min', ascending = False).astype(int)
output['this_month'] = dt.datetime.strptime(this_month, '%Y-%m').strftime('%B %Y')

######################################################################################################################