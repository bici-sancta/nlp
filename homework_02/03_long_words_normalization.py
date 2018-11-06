
# coding: utf-8

# In[261]:

#
# ... file : longest_words.py
#
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...
# ... msds 7337 NLP
# ... homework 02
# ... gutenberg - documment vocabulary normalization
# ... pmcdevitt@smu.edu
# ... 15-sep-2018
# ...
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... load packages
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import ConditionalFreqDist

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rc('xtick', labelsize=20)     
plt.rc('ytick', labelsize=20)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... some directory and file name definitions
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

files = ".*\.txt"
home_dir = "/home/mcdevitt/_ds/_smu/_src/nlp/homework_02/"
corpus_root = "./text/"
corpus_clean = "./text_no_license/"
plot_dir = "./plots/"

os.chdir(home_dir)


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... long_words characterizations
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def long_words(text_input) :    
    text_unique = set(text_input)    
    long_words = [w for w in text_unique if len(w) > 7]
    longest = sorted(long_words, key = len, reverse = True)[0]
    return longest
    

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... remove all non-alpha characters
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def clean_text(start_text) :
    text = [re.sub('^((?![a-zA-Z ]).)*$', '', x) for x in start_text]
    text = [re.sub('_', '', x) for x in text]
    text = [x.lower() for x in text]
    text = [x for x in text if x != '']
    return text


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... read in text with gutenberg license removed / assemble corpus for evaluation
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

readers = PlaintextCorpusReader(corpus_clean, files)

files = readers.fileids()
files[0:10]

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... create table to accumulate summary data
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results_tbl = pd.DataFrame(columns =
    ['text_name',
     'long_mot',
     'long_word_length'])

i_index = []
i_index = 0

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... loop thru each text to assemble metrics
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

print("Some basic statistics\n")

for fileid in readers.fileids():
    rtxt = clean_text(readers.words(fileid))
    
    le_mot_le_plus_long = long_words(rtxt)
    print("\n", i_index, "--", fileid, " : ", le_mot_le_plus_long, "\n")

    table_data = {
     'text_name' : fileid,
     'long_mot' : le_mot_le_plus_long,
     'long_word_length' : len(le_mot_le_plus_long)
    } 

    df_tbl = pd.DataFrame(table_data,
        columns = ['text_name',
             'long_mot', 'long_word_length'],
    index = [i_index + 1])
    i_index += 1
    results_tbl = results_tbl.append(df_tbl)
    
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... basic statistics table complete
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results_tbl = results_tbl.sort_values(results_tbl.columns[2], ascending = False)
results_tbl

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... normalize each column by max column value
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results_nrmlzd = results_tbl.copy()
results_nrmlzd = results_nrmlzd.sort_values(results_nrmlzd.columns[2], ascending = False)

df = results_nrmlzd.iloc[:, 2]
df_nrml = df / df.max()

df_labels = results_nrmlzd.iloc[:, 0]
df_words =  results_nrmlzd.iloc[:, 1]

results_nrmlzd = pd.concat([df_labels, df_words, df_nrml], axis = 1)

print('results_nrmlzd - ')

results_nrmlzd = results_nrmlzd.sort_values(results_nrmlzd.columns[2], ascending = False)
print('results_nrmlzd - sorted by col num_vocab')
results_nrmlzd

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... add some shorter title names to fit within plotting space
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results_tbl['title_short'] = [txt[:25] for txt in results_tbl['text_name']]
results_tbl['title_short'][:5]

results_nrmlzd['title_short'] = [txt[:25] for txt in results_nrmlzd['text_name']]
results_nrmlzd['title_short'][:5]

results_nrmlzd = results_nrmlzd.sort_values(results_nrmlzd.columns[2], ascending = False)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... plot - max word length normalized - sorted in descending order
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

N = len(results_nrmlzd)
ind = np.arange(N) 
width = 0.5

_ = plt.figure(figsize = (18, 12))
offset = 0
_ = plt.bar(ind + offset, results_nrmlzd['long_word_length'],
            width,
            label='Longest word',
            color = 'slateblue')

_ = plt.xticks(ind + width / 2, results_nrmlzd['title_short'], fontsize = '15')
_ = plt.xticks(rotation=90)
_ = plt.legend(loc='upper left')
_ = plt.title('Word Lengths - 100+ Readers - Longest Word (normalized)', fontsize = '30')

_ = axes = plt.gca()
_ = axes.set_ylim([0.4, 1])

_ = plt.savefig(plot_dir + 'childrens_books_normalized_longword.png')
_ = plt.show()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... write table to output file for future reference
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

os.chdir(home_dir)

file_name = "max_word_length_normalized.txt"

results_nrmlzd.to_csv(file_name,
                      header = True,
                      index = None,
                      sep=',',
                      mode = 'a')
    
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... end_of_file
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    

