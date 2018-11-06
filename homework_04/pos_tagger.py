
# coding: utf-8

# In[6]:

#
# ... file : pos_tagger.py
#
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...
# ... msds 7337 NLP
# ... homework 04
# ... part of speech tagging
# ... pmcdevitt@smu.edu
# ... 10-oct-2018
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
import pattern
from pattern.en import tag


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

home_dir = "/home/mcdevitt/_ds/_smu/_src/nlp/homework_04/"
corpus_root = "./text/"
plot_dir = "./plots/"
files = ".*\.txt"

os.chdir(home_dir)
os.chdir(corpus_root)

penn_pos = pd.read_csv("penn_treebank.csv")


file = open("two_sentences.txt", "r")
for line in file :
    
    print (line)

    tokens = nltk.word_tokenize(line)

# ... nltk pos tagger

    tag_nltk = nltk.pos_tag(tokens)

    df_tag_nltk = pd.DataFrame(tag_nltk, columns = ['word', 'pos'])
    df_dscr_nltk = pd.merge(df_tag_nltk, penn_pos, on='pos', how='left')
    df_dscr_nltk.columns = ['word', 'pos_nltk', 'descr_nltk']
    
# ... pattern pos tagger

    tag_ptrn = tag(line)
    
    df_tag_ptrn = pd.DataFrame(tag_ptrn, columns = ['word', 'pos'])
    df_dscr_ptrn = pd.merge(df_tag_ptrn, penn_pos, on = 'pos', how = 'left')
    df_dscr_ptrn.columns = ['word', 'pos_pattern', 'descr_pattern']
    
    df_tag_all = pd.concat([df_dscr_nltk, df_dscr_ptrn], axis = 1)
    df_tag_all


# In[ ]:



