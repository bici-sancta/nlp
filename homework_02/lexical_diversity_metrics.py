
# coding: utf-8

# In[34]:

#
# ... file : lexical_diversity_metrics.py
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
# ... read in saved .txt files (.csv) from normalizations
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

file_wordl = 'max_word_length_normalized.csv'
file_vocab = 'vocab_normalized.csv'

selected_texts = [
    "mcguffey's_first_eclectic_reader,_revised_edition.txt", 
    "mcguffey's_second_eclectic_reader.txt", 
    "mcguffey's_third_eclectic_reader.txt", 
    "mcguffey's_fourth_eclectic_reader.txt", 
    "mcguffey's_fifth_eclectic_reader.txt", 
    "mcguffey's_sixth_eclectic_reader.txt",
    "new_national_first_reader",
    "the_ontario_high_school_reader.txt",
    "the_literary_world_seventh_reader.txt"
]

df_wordl = pd.read_csv(file_wordl)
df_vocab = pd.read_csv(file_vocab)

df_vocab[:10]

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... condense to normalized metrics and selected texts
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_vocab_subset = df_vocab.loc[df_vocab['text_name'].isin(selected_texts)]
df_wordl_subset = df_wordl.loc[df_wordl['text_name'].isin(selected_texts)]

df_wordl_subset = df_wordl_subset[['text_name','long_word_length']]
df_vocab_subset = df_vocab_subset[['text_name', 'title_short', 'num_vocab', 'lex_div']]

df_wordl_subset = df_wordl_subset.sort_values(df_wordl_subset.columns[0])
df_vocab_subset = df_vocab_subset.sort_values(df_vocab_subset.columns[0])

df_wordl_subset = df_wordl_subset.reset_index(drop=True)
df_vocab_subset = df_vocab_subset.reset_index(drop=True)

df_wordl_subset = df_wordl_subset[['long_word_length']]

df_metrics = pd.concat([df_vocab_subset, df_wordl_subset], axis = 1)
df_metrics.columns = ['text_name', 'title_short', 'vocab_nrml', 'lex_div_nrml', 'word_length_nrml']

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... combine all 3 normalized metrics to total score
# ... - use addition (sum_scores) and multiplication (mlt_scores)
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_metrics['sum_scores'] = df_metrics['vocab_nrml'] + df_metrics['lex_div_nrml'] + df_metrics['word_length_nrml']
df_metrics['mlt_scores'] = df_metrics['vocab_nrml'] * df_metrics['lex_div_nrml'] * df_metrics['word_length_nrml']

df_metrics = df_metrics.sort_values(df_metrics.columns[5], ascending = False)
df_metrics
df_metrics = df_metrics.sort_values(df_metrics.columns[6], ascending = False)
df_metrics

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... plot comparison of all (normalized) metrics - 
# ... sorted in descending mlt_scores order
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

N = len(df_metrics)

ind = np.arange(N) 
width = 0.18

_ = plt.figure(figsize = (18, 12))

offset = width / 2
offset = 0
_ = plt.bar(ind + offset, df_metrics['lex_div_nrml'],
            width, label='Lex_Div', color = 'tomato')
_ = plt.bar(ind + offset + width, df_metrics['vocab_nrml'],
            width, label='Vocab', color = 'dodgerblue', alpha = 0.9)
_ = plt.bar(ind + offset + width*2, df_metrics['word_length_nrml'],
            width, label='Word Length', color = 'slateblue', alpha = 0.9)

#_ = plt.bar(ind + offset + width*3, df_metrics['sum_scores'], width, label='Sum Scores', color = 'orchid', alpha = 0.9)

_ = plt.bar(ind + offset + width*4, df_metrics['mlt_scores']*4,
            width,
            label='Mult Scores',
            color = 'darkolivegreen',
            alpha = 0.9)

_ = plt.xticks(ind + width / 2, df_metrics['title_short'])
_ = plt.xticks(rotation=90)
_ = plt.legend(loc='upper right')
_ = plt.title('Normalized Characteristics Comparison', fontsize = '30')

axes = plt.gca()
axes.set_ylim([0, 1.2])

_ = plt.savefig(plot_dir + 'lex_div_normalized_scores.png')
_ = plt.show()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... end_of_file
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

