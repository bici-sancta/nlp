
# coding: utf-8

# In[68]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... report platform
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import platform; print platform.platform()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... load packages
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import os
from os import path

import sys; print "Python", sys.version
reload(sys)
sys.setdefaultencoding('utf8')
sys.stdout.flush()

import timeit
import time
import re; print "re", re.__version__

import numpy as np
import pandas as pd
import statistics

import warnings
warnings.filterwarnings("ignore")

import string
import unicodedata
import pattern
import collections

from tabulate import tabulate

from tqdm import tqdm

import nltk; print "nltk", nltk.__version__
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import ConditionalFreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig

import mglearn; print "mglearn", mglearn.__version__

from PIL import Image
from wordcloud import WordCloud,STOPWORDS

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... add plot utilities
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use('seaborn-whitegrid')

rcParams.update({'figure.autolayout': True})
plt.rc('xtick', labelsize=20)     
plt.rc('ytick', labelsize=20)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... support importing other ipynb notebooks
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import import_ipynb

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... set some options
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

get_ipython().magic(u'pylab')
get_ipython().magic(u'matplotlib inline')
pd.set_option('max_colwidth',400)


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... include some local utility functions 
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def printf(format, *args):
    sys.stdout.write(format % args)

def min_word_length(text, min_length = 2):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if len(token) > min_length]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# ref : https://stackoverflow.com/questions/3861674/split-string-by-number-of-words-with-python
def group_words(s, n):
    words = s.split()
    for i in xrange(0, len(words), n):
        yield ' '.join(words[i:i+n])

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... character cleaning functions
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def clean_text(start_text) :
    text = [re.sub('^((?![a-zA-Z ]).)*$', '', x) for x in start_text]
    text = [re.sub('_', '', x) for x in text]
    text = [x.lower() for x in text]
    text = [x for x in text if x != '']
    return text


def remove_non_ascii(text):
    L = [32, 44, 46, 65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,
         90,97,98,99,100,101,102,103, 104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,
         120,121,122]
    text = str(text)
    return ''.join(i for i in text if ord(i) in L)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... print topic features
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def selected_topics(model, vectorizer, top_n=10):
    
    for idx, topic in enumerate(model.components_):
        
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... import Sarkar functions : https://github.com/dipanjanS/text-analytics-with-python
# ...
# ... (functions modified locally for specific functionality updates)
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

from normalization_ameliorer import *
from featurizer import *
from cluster_utilities import *

import normalization_ameliorer
import featurizer
import cluster_utilities

reload (normalization_ameliorer)
reload (featurizer)
reload (cluster_utilities)


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... declare some directory locations
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

file_mask = ".*\.txt"

home_dir = "/home/mcdevitt/_ds/_smu/_src/nlp/topic_model/"
corpus_dir = "./corpus/"
corpus_clean = "./corpus_no_license/"

plot_dir = "./plots/"

os.chdir(home_dir)


# In[20]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... read in text with gutenberg license removed / assemble corpus for evaluation
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

readers = PlaintextCorpusReader(corpus_clean, file_mask)

bookname = readers.fileids()
bookname[0:10]


# In[23]:


readers.raw(bookname[99])


# In[24]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... create table for collecting results
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_book  = pd.DataFrame(columns = [
    'title',
    'length',
    'text'])

i_index=[]
i_index = 0

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

for fileid in readers.fileids():
    rtxt = readers.raw(fileid)
    
    dict_this = {'title' : fileid,
        'length' : len(rtxt),
        'text' : rtxt}
    df_this = pd.DataFrame(dict_this, index = [i_index + 1])
    df_book = df_book.append(df_this)
    
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... reset index of data frame 
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_book.reset_index(drop = True, inplace = True)


# In[25]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... save data frame for later recall 
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_book.to_pickle('df_book.pkl')


# In[26]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... recover books data frame 
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_book = pd.read_pickle('df_book.pkl')

sum(df_book['length'])

df_book.head()
df_book.tail()


# In[27]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... collector table for chunks of books
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_chunk  = pd.DataFrame(columns = [
    'title',
    'chunk',
    'length',
    'text'])

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... collector table for processing info during normalization
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_norm  = pd.DataFrame(columns = [
    'title',
    'len_start',
    'len_end',
    'time',
    'ratio',
    'rate',
    'clock_time',
    'process_times'])

i_index=[]
i_index = 0

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... just to make sure stopword list has no duplicates
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

stopword_set = list(set(stopword_list))

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... set some initial conditions prior to normalization 
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_select = df_book

# ... select subset of texts to evaluate exeution cycle in shorter time
#df_select = df_book[0:10]

normalized_corpus = []

itxt = 0
i_index = 0

chunk_size = 2000

df_select


# In[28]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... normalization 
# ...
# ... this section of code requires ~2 hours process time on
# ...    intel i5 @ 1.7Ghzx4, linux ubuntu 16.04lts
# ...              !!!!!!!!!!!!!!!!!!!
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



for text, title in zip(df_select['text'], df_select['title']) :
    
    printf("\n\n%40s\n", title)

    start_time = timeit.default_timer()
    txt_start = len(text)
    
# expand contractions

    text = expand_contractions(text, CONTRACTION_MAP)
    
    t1 = timeit.default_timer() - start_time
    s1 = len(text)
    
# ... pos tag, then downselect to only NOUNs

    pos_tagged_text = pos_tag_text(text)
    t2 = timeit.default_timer() - start_time
    
    pos_selected_tokens = [word if (pos_tag == 'n') else '.'
                            for word, pos_tag in pos_tagged_text]    
    t3 = timeit.default_timer() - start_time

    s2 = len(pos_selected_tokens)

# ... lemmatize

    lemmatized_tokens = [wnl.lemmatize(word, 'n')                     
                             for word in pos_selected_tokens]
    
    s3 = len(lemmatized_tokens)
    
    lemmatized_text = ' '.join(lemmatized_tokens)
    t4 = timeit.default_timer() - start_time
    s4 = len(lemmatized_text)
    
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# .. chunk each text into multiple same-sized chunks of chunk_size
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    chunks = list(group_words(lemmatized_text, chunk_size))
    ichunk = 0
    
    for chunk in chunks :
        
# .. remove special characters
    
        text = remove_special_characters(chunk)

# .. remove 1 and 2 letter tokens

        text = min_word_length(text)
        
# .. remove stop words
# ... ref : https://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python/19560841

        pattern = re.compile(r'\b(' + r'|'.join(stopword_set) + r')\b\s*')
        text = pattern.sub('', text)

        dict_this = {'title' : title,
                     'chunk' : ichunk,
                     'length' : len(text),
                     'text' : text}
        
        df_this = pd.DataFrame(dict_this, index = [i_index + 1])
        df_chunk = df_chunk.append(df_this)
        ichunk = ichunk + 1
        
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... add next chunk to normalized corpus ... 
# ...     ---> this is the corpus for topic model
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        normalized_corpus.append(text)

    t5 = timeit.default_timer() - start_time
    s5 = len(text)

    txt_end = s5

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... store some information about processing times, text lengths
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    elapsed = timeit.default_timer() - start_time
    
    printf("%d | %d\n", len(pos_selected_tokens), len(lemmatized_tokens))
    printf(4*"%7d | " + "%7d\n", s1, s2, s3, s4, s5)
    
    printf("%3d : %8d | %8d | %.1f | %.2f | %.1f | ",
               itxt, txt_start, txt_end, (float(txt_end)/float(txt_start)*100.0),
               elapsed, len(text) / elapsed)
    itxt = itxt + 1

    now = time.time()
    printf ("%02d:%02d:%02d |",
                         time.localtime(now).tm_hour,
                         time.localtime(now).tm_min,
                         time.localtime(now).tm_sec)
    printf(4*"%.1f, " + "%.1f", t1, t2-t1, t3-t2, t4-t3, t5-t4)
    print "\n"
    print '-'*80
    
    clock_time = "%02d:%02d:%02d |" % (time.localtime(now).tm_hour,
                             time.localtime(now).tm_min,
                             time.localtime(now).tm_sec)
    process_times = "%.1f, %.1f, %.1f, %.1f, %.1f" % (t1, t2-t1, t3-t2, t4-t3, t5-t4)
    
    dict_this = {'title' : title,
                'len_start' : txt_start,
                'len_end' : txt_end,
                'time' : elapsed,
                'ratio' : float(txt_end)/float(txt_start),
                'rate' : len(text) / elapsed,
                'clock_time' : clock_time,
                'process_times' : process_times}
    
    df_this = pd.DataFrame(dict_this, index = [i_index + 1])
    df_norm = df_norm.append(df_this)
    
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
print '~'*80

df_norm


# In[29]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... save chunked data frame & normalized corpus list for later recall 
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import pickle

df_chunk.to_pickle('df_chunk.pkl')

with open('normalized_corpus.pkl', 'wb') as fp:
    pickle.dump(normalized_corpus, fp)


# In[49]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... read in chunked data frame from .pkl file
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_chunk = pd.read_pickle('df_chunk.pkl')


# In[53]:


df_chunk = df_chunk.reset_index(drop = True)

df_chunk


# In[81]:





# In[32]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... recall normalized corpus from disk
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

with open ('normalized_corpus.pkl', 'rb') as fp:
    normalized_corpus = pickle.load(fp)

len(normalized_corpus)
normalized_corpus[0:200]


# In[33]:



# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... build feature matrix from normalized corpus
# ...
# ... this builds 2 sets ... one based on token counts, 2nd based on tfidf
# ... later analysis shows not much difference for this data set,
# ... so the basic counts is retained for this topic model
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

start_time = timeit.default_timer()

min_df = 0.1
max_df = 0.5
ngram_range = (1, 1)

vector_count = CountVectorizer(binary = False, min_df = min_df,
                                     max_df = max_df, ngram_range = ngram_range)

vector_tfidf = TfidfVectorizer(min_df = min_df, max_df = max_df, 
                                     ngram_range = ngram_range)

feature_matrix_count = vector_count.fit_transform(normalized_corpus).astype(float)
feature_matrix_tfidf = vector_tfidf.fit_transform(normalized_corpus).astype(float)

feature_names_count = vector_count.get_feature_names()
feature_names_tfidf = vector_tfidf.get_feature_names()

elapsed = timeit.default_timer() - start_time
print elapsed



# In[34]:



# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... Topic Model --- Latent Dirichlet Allocation Model
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

NUM_TOPICS = 9


# In[37]:


get_ipython().run_cell_magic(u'time', u'', u"\nlda = LatentDirichletAllocation(n_components = NUM_TOPICS,\n                                max_iter = 10,\n                                learning_method = 'online',\n                                verbose = True)\n\ndata_lda_tfidf = lda.fit_transform(feature_matrix_tfidf)\ndata_lda_count = lda.fit_transform(feature_matrix_count)")


# In[119]:



type(data_lda_count)

data_lda_count[[0,1,2], :]


# In[113]:


type(data_lda_count)

data_lda_count.shape

data_lda_count[0:10]


# In[161]:


#titles = df_chunk['title'].unique()

df_book_topic  = pd.DataFrame(columns = [
    'title',
    'topic',
    'score',
    'list'])
i_index=[]
i_index = 0

len(titles)

#titles

#titles[0:2]

#df_chunk.loc[df_chunk['title'].isin(titles[0:2])].index.values.tolist()
ii = 0
indx = []
yyy = []
A = []

for t in titles :
#    print ii
#    print t
#    sys.stdout.flush()
    indx = df_chunk.loc[df_chunk['title'] == titles[ii]].index.values.tolist()
    t
    xxx = data_lda_count[indx, :]
#    xxx
    vsum = np.sum(xxx, axis = 0)
    mag = np.sqrt(vsum.dot(vsum))
    vsum
    mag
    unit_vec = vsum / mag
    unit_vec
    lda_topic = np.argmax(unit_vec)
    A.append(unit_vec)
    ii = ii + 1
    
    dict_this = {'title' : t,
                'topic' : lda_topic,
                'score' : unit_vec[np.argmax(unit_vec)]}   
#               'list' : (unit_vec)]}

    dict_this['list'] = unit_vec
    dict_this
    
    df_this = pd.DataFrame(dict_this, index = [i_index + 1])
    i_index = i_index + 1
    
    df_book_topic = df_book_topic.append(df_this)

#    print indx
    
len(indx)

#for p in indx: print p

len(A)
type(A)

A[2]

Array = np.array(A)

type(Array)

Array[[0,1,2], :]

Array.shape



# In[154]:



df_book_topic['topic'].value_counts()

df_book_topic.sort_values(['topic', 'score'], ascending=[True, False])


# In[143]:


xxx = list(Array[[0],:])

type(xxx)

yyy = xxx.sort()

yyy


# In[96]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... keywords for topics clustered by Latent Dirichlet Allocation
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

print("LDA Model (tfidf):")
selected_topics(lda, vector_tfidf)

print("LDA Model (count):")
selected_topics(lda, vector_count)


# In[21]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... Check perplexity
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

data_lda_tfidf_prplx = lda.perplexity(feature_matrix_tfidf)
data_lda_count_prplx = lda.perplexity(feature_matrix_count)

data_lda_tfidf_prplx
data_lda_count_prplx


# ****
# 
# ### Visualizing LDA results with pyLDAvis
# 
# ***
# 
# 1. Topics are on the left while keywords are displayed on the right
# 2. More prevalent topics are identified with larger topic circles
# 3. The spatial relationship (distance) between topics indicates relative similarity / dissimilarity
# 4. Keywords are displayed in descending order of relevance to the selected topic
# 5. lamba value for relevance includes / excludes words with increasing / decreasing generalizabilty to other topcis
#     - low lamba values : words are specific to selected topic
#     - higher lambda value : words are associated to more (all) topics
# 

# In[ ]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... Visualization with term COUNTS model
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda, feature_matrix_count, vector_count, mds = 'tsne')

dash


# In[ ]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... Visualization with term TF-IDF model
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda, feature_matrix_tfidf, vector_tfidf, mds = 'tsne')

dash


# In[155]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... Print each topic and associated Top 10 words
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

sorting = np.argsort(lda.components_)[:,::-1]
features = np.array(vector_count.get_feature_names())

mglearn.tools.print_topics(topics = range(9), feature_names = features,
                            sorting = sorting, topics_per_chunk = 10, n_words = 10)


# 
# ```
# topic 0       topic 1       topic 2       topic 3       topic 4       topic 5       topic 6       topic 7       topic 8       
# --------      --------      --------      --------      --------      --------      --------      --------      --------      
# horse         sir           sea           lord          room          state         father        night         god           
# war           horse         water         god           face          power         woman         boy           soul          
# army          sword         ship          land          girl          law           wife          door          world         
# soldier       name          captain       city          door          government    child         arm           heaven        
# officer       london        foot          child         woman         case          brother       side          death         
# arm           brother       tree          nation        friend        person        mother        round         spirit        
# field         battle        year          servant       mother        subject       year          foot          earth         
# force         lord          land          battle        gentleman     nature        daughter      light         friend        
# order         arm           air           court         table         opinion       master        bed           truth         
# road          death         wind          peace         voice         question      family        street        body          
# ```

# In[116]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... Do the Word Cloud
# ...
# ... for this iteration ... just add in all the tokens in the total
# ... normalized (NOUNs only) corpus
# ...
# ... for future iterations ...
# ...    consider creating word cloud specific for each topic and
# ...    each book, for customer/participant engagment
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

normalized_text = ' '.join(normalized_corpus)

# ... add in a bunch of 'guttenberg's to advertise their site in the cloud

gut1000 = ' '.join(['guttenberg'] * 20000)
wc_text = gut1000 + normalized_text

# ... and the words in the cloud

wc = WordCloud(background_color = "#050540",
               colormap="Spectral",
               width=1600,
               height=900,
               max_words = 1000,
               collocations = False,
#               stopwords = ['hand']
              )

wc.generate(wc_text)

plt.figure(figsize = (16, 9))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis("off")

savefig('wordcloud.2.png')


# In[26]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... end_of_file
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# In[27]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... environment and package versions
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

print('\n')
print("_"*70)
print('The environment and package versions :')
print('\n')

import platform
import os
import sys
import pandas as pd
import numpy as np
import bs4
from bs4 import BeautifulSoup
import re
import sklearn
import matplotlib
import mglearn

print(platform.platform())
print('Python', sys.version)
print("pandas version:", pd.__version__)
print('OS', os.name)
print('Numpy', np.__version__)
print('Beautiful Soup', bs4.__version__)
print('Regex', re.__version__)
print('scikit-learn version', sklearn.__version__)
print('matplotlib', matplotlib.__version__)
print("mglearn version:", mglearn.__version__)

print('\n')
print("~"*70)
print('\n')

