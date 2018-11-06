
# coding: utf-8

# In[44]:

#
# ... file : vocabulary_size_normalization.py
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
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import PlaintextCorpusReader

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
plot_dir = "./plots/"

os.chdir(home_dir)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... lexical diversity score - as given in nltk site
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def lexical_diversity(my_text_data):
    tokens = len(my_text_data)
    types = len(set(my_text_data))
    diversity_score = types / tokens
    return diversity_score

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... read in texts / assemble corpus for evaluation
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

readers = PlaintextCorpusReader(corpus_root, files)

files = readers.fileids()
files[0:10]

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... create table to accumulate summary data
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results_tbl = pd.DataFrame(columns =
    ['text_name',
     'num_chars',
     'num_words',
     'num_sents',
     'num_vocab',
     'tokens',
     'types',
     'lex_div'])

i_index = []
i_index = 0

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... loop thru each text to assemble metrics
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

print("Some basic statistics\n")

for fileid in readers.fileids():
    
    print(i_index, "---", fileid)
    
    num_chars = len(readers.raw(fileid))
    num_words = len(readers.words(fileid))
    num_sents = len(readers.sents(fileid))
    tokens = len(readers.words(fileid))
    types = len(set(readers.words(fileid)))
    
    num_vocab = len(set(w.lower() for w in readers.words(fileid)))

    rtxt = readers.words(fileid)
    ldiv = lexical_diversity(rtxt)

    table_data = {
     'text_name' : fileid,
     'num_chars' : num_chars,
     'num_words' : num_words,
     'num_sents' : num_sents,
     'num_vocab' : num_vocab,
     'tokens' : tokens,
     'types' : types,
     'lex_div' : ldiv
    } 

    df_tbl = pd.DataFrame(table_data,
        columns = ['text_name',
             'num_chars',
             'num_words',
             'num_sents',
             'num_vocab',
             'tokens',
             'types',
             'lex_div'],
    index = [i_index + 1])
    i_index += 1
    results_tbl = results_tbl.append(df_tbl)
    
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... basic statistics table complete
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results_tbl = results_tbl.sort_values(results_tbl.columns[4], ascending = False)
print('Results_tbl - sorted by col num_vocab')
results_tbl

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... normalize each column by max column value
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results_nrmlzd = results_tbl.copy()
results_nrmlzd = results_nrmlzd.sort_values(results_nrmlzd.columns[4], ascending = False)

df = results_nrmlzd.iloc[:, 1:8]
df_nrml = df / df.max()

df_labels = results_nrmlzd.iloc[:, 0]

results_nrmlzd = pd.concat([df_labels, df_nrml], axis = 1)

results_nrmlzd['vocab_ldiv'] = results_nrmlzd.apply(lambda x: x.lex_div * (x.num_words), axis=1)
results_nrmlzd['vocab_ldiv'] = results_nrmlzd['vocab_ldiv'] / results_nrmlzd['vocab_ldiv'].max()

print('results_nrmlzd - ')

results_nrmlzd = results_nrmlzd.sort_values(results_nrmlzd.columns[4], ascending = False)
print('results_nrmlzd - sorted by col num_vocab')
results_nrmlzd

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... add some shorter title names to fit within plotting space
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results_tbl['title_short'] = [txt[:25] for txt in results_tbl['text_name']]
results_tbl['title_short'][:5]

results_nrmlzd['title_short'] = [txt[:25] for txt in results_nrmlzd['text_name']]
results_nrmlzd['title_short'][:5]

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... plot - vocabulary normalized size - sorted in descending order
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

N = len(results_nrmlzd)

ind = np.arange(N) 
width = 0.5

_ = plt.figure(figsize = (18, 12))
offset = 0
_ = plt.bar(ind + offset, results_nrmlzd['num_vocab'], width, label='Vocab', color = 'orchid')

_ = plt.xticks(ind + width / 2, results_nrmlzd['title_short'], fontsize = '15')
_ = plt.xticks(rotation=90, )
_ = plt.legend(loc='upper left')
_ = plt.title('Vocabulary Size Normalized - 100+ Childrens Books', fontsize = '30')

axes = plt.gca()
_ = plt.savefig(plot_dir + 'childrens_books_normalized_vocab.png')
_ = plt.show()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... write table to output file for future reference
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

os.chdir(home_dir)

file_name = "vocab_normalized.txt"

results_nrmlzd.to_csv(file_name,
                      header = True,
                      index = None,
                      sep=',',
                      mode = 'a')
    
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... end_of_file
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    

