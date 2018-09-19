
#
# ... file : collect_texts_from_gutenberg.py
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
import string
import strip

import nltk
from nltk.corpus import PlaintextCorpusReader

from lxml import html
import requests

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... some plotting defaults
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rc('xtick', labelsize=20)     
plt.rc('ytick', labelsize=20)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... declare some directory locations
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

home_dir = "/home/mcdevitt/_ds/_smu/_src/nlp/homework_02/"
corpus_dir = "./text/"
plot_dir = "./plots/"

os.chdir(home_dir)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... extract urls for targeted texts 
# ...
# ... if link is in this format : //www.gutenberg.org/ebooks/7841
# ...
# ... then texts is at this corresponding url : http://www.gutenberg.org/cache/epub/7841/pg7841.txt
# ... or at this url :                          http://www.gutenberg.org/files/7841-0.txt
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

page = requests.get('http://www.gutenberg.org/wiki/Children%27s_Instructional_Books_(Bookshelf)')
tree = html.fromstring(page.content)

links = tree.xpath('//a/@href')

book_link_mask = 'gutenberg.org/ebooks/'
book_links = [s for s in links if book_link_mask.lower() in s.lower()]
book_links = [lnk.replace('ebooks', 'cache/epub') for lnk in book_links]

# ... extract page number from end of url

pg_num = [lnk.split("epub/",1)[1] for lnk in book_links]
pg_num[0:5]

# ... recombine to create full url

text_link = ['http:' + lnk + '/pg' + pg + '.txt' for lnk,pg in zip(book_links, pg_num)]
text_link[0:5]

len(text_link)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... extract book titles 
# ... tree.xpath('//div[@title="buyer-name"]/text()')
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

title = []

for ib in range(0, len(pg_num)) :
    path = '//a[@title="ebook:' + pg_num[ib] + '"]/text()'
    nxt_title = tree.xpath(path)[0]
    title.append(nxt_title)

title[0:5]
len(title)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... clean up titles so filenames have standard chars and no spaces
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

''.join([x if x in string.printable else '' for x in title])

title = [x.replace('\n', '') for x in title]
title = [x.replace(' ', '_') for x in title]
title = [x.lower() for x in title]

title[0:10]

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... download texts to local directory
# ... capture errors to identify special case urls
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import wget

print('Beginning file downloads ...')

os.chdir(home_dir)
os.chdir(corpus_dir)

error_indx = []
pipe = ' | '

for ib in range(0, len(pg_num)) : 
    
    url = text_link[ib]
    file_name = title[ib] + '.txt'
    
    try :
        wget.download(url, file_name)
        
    except HTTPError as e:
        print ("HTTP error({0}): {1}".format(e.errno, e.strerror))
        error_indx.append(str(ib) + pipe + pg_num[ib] + pipe + title[ib])
        
        try :
            fix_url = text_link[ib].replace('.txt', '-0.txt')
            fix_url = fix.replace('/pg', '/')
            fix_url = fix.replace('/cache/epub/', '/files/')
            print(fix_url)
            wget.download(fix_url, file_name)
            
        except :
            print ('attempted fix_url did not resolve error')
            
print('... file downloads complete.')

error_indx

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... end_of_file
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

