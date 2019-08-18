
# coding: utf-8

# In[1]:


#
# ... file : remove_gutenberg_license.py
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


# In[2]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... some directory and file name definitions
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

files = ".*\.txt"
home_dir = "/home/mcdevitt/_ds/_smu/_src/nlp/homework_08/"
corpus_root = "./corpus/"
corpus_clean = "./corpus_no_license/"
plot_dir = "./plots/"

os.chdir(home_dir)


# In[7]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... remove gutenberg license function
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def remove_license(text_raw) :
    
    license_head1 = '*** START OF THIS PROJECT GUTENBERG EBOOK'
    license_head2 = '*** START OF THE PROJECT GUTENBERG EBOOK'
    license_head3 = '***START OF THE PROJECT GUTENBERG EBOOK'
    
    license_tail1 = 'End of the Project Gutenberg EBook'
    license_tail2 = 'End of Project Gutenberg'
    license_tail3 = '***END OF THE PROJECT GUTENBERG EBOOK'
    
    try :
        book_tail_text = text_raw.split(license_head1, 1)[1]
    except IndexError:
        try :
            book_tail_text = text_raw.split(license_head2, 1)[1]
        except IndexError:
            book_tail_text = text_raw.split(license_head3, 1)[1]
    
    text_without_license = book_tail_text.split(license_tail1, 1)[0]
    
    if(len(text_without_license) == len(book_tail_text)) :
        text_without_license = book_tail_text.split(license_tail2, 1)[0]
    
    if(len(text_without_license) == len(book_tail_text)) :
        text_without_license = book_tail_text.split(license_tail3, 1)[0]

    return text_without_license.encode('utf-8')


# In[4]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... read in raw downloaded texts / assemble corpus for evaluation
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

readers = PlaintextCorpusReader(corpus_root, files)

files = readers.fileids()
files[0:10]


# In[10]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... remove license and write cleaned version to new directory
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

for fileid in readers.fileids():
    print fileid
    
    try :
        btxt = remove_license(readers.raw(fileid))
        text_file = open(corpus_clean + fileid, "w")
        text_file.write(btxt)
        text_file.close()
        
    except :
        print fileid
        print "License removal not successful"
        print '~'*80
        
        
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... end_of_file
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    

