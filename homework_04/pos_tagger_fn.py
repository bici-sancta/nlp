
# coding: utf-8

# In[10]:

import pandas as pd
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import ConditionalFreqDist
from pattern.en import tag

def pos_tagger (phrase, tagger) : 
    
    penn_pos = pd.read_csv("penn_treebank.csv")

    if tagger == 'nltk' :

# ... nltk pos tagger

        tokens = nltk.word_tokenize(phrase)
        tag_nltk = nltk.pos_tag(tokens)

        df_tag = pd.DataFrame(tag_nltk, columns = ['word', 'pos'])
        df_dscr = pd.merge(df_tag, penn_pos, on = 'pos', how = 'left')
        df_dscr.columns = ['word', 'pos', 'descr']
 
    elif tagger == 'pattern': 

# ... pattern pos tagger

        tag_ptrn = tag(line)

        df_tag = pd.DataFrame(tag_ptrn, columns = ['word', 'pos'])
        df_dscr = pd.merge(df_tag, penn_pos, on = 'pos', how = 'left')
        df_dscr.columns = ['word', 'pos', 'descr']
            
    return (df_dscr)

