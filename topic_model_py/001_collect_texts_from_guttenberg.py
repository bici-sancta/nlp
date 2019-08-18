
# coding: utf-8

# In[2]:


#
# ... file : collect_texts_from_gutenberg
#
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...
# ... msds 7337 NLP
# ... pmcdevitt@smu.edu
# ... 29-nov-2018
# ...
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# In[81]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... necessary packages for Ben Brock
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import platform; print platform.platform()
import sys; print "Python", sys.version
import nltk; print "nltk", nltk.__version__
#from bs4 import BeautifulSoup, SoupStrainer
import requests; print "requests", requests.__version__

try :
    from urllib2 import Request, urlopen, HTTPError
except :
    from urllib.request import Request, urlopen
    
import re; print "re", re.__version__

from pattern.en import parsetree

import os
#print (os.environ['CONDA_DEFAULT_ENV'])


# In[66]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... declare some directory locations
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

home_dir = "/home/mcdevitt/_ds/_smu/_src/nlp/homework_08/"
corpus_dir = "./corpus/"
plot_dir = "./plots/"

os.chdir(home_dir)


# In[45]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... extract urls for targeted texts 
# ...
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

base_url = 'https://www.gutenberg.org'

page = requests.get(base_url + '/browse/scores/top#books-last30')
tree = html.fromstring(page.content)

links = tree.xpath('//a/@href')

links[0:5]

book_link_mask = '/ebooks/'
book_link = [s for s in links if book_link_mask.lower() in s.lower()]

book_link = book_link[1:]
unique_book_link = list(set(book_link))

file_link = [lnk.replace('ebooks', 'files') for lnk in unique_book_link]

unique_book_link[0:5]
file_link[0:5]

# ... extract page number from end of url

pg_num = [lnk.split("files/",1)[1] for lnk in file_link]
pg_num[0:5]

# ... recombine to create full url

text_link = [base_url + lnk + '.txt' for lnk in file_link]
text_link[0:5]

unique_book_link[0:5]
pg_num[0:5]
text_link[0:5]

len(unique_book_link)
len(pg_num)
len(text_link)

text_link[100:]


# In[46]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... extract book titles
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

title = []

for ib in range(0, len(pg_num)) :
    page = base_url + unique_book_link[ib]
    print page
    
    source = requests.get(page)
    soup = BeautifulSoup(source.content, "lxml")
    
    nxt_title = soup.select('h1[itemprop=name]')[0].text
    print nxt_title
    
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


# In[83]:


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
    
    url = base_url + unique_book_link[ib]
    print url
    
    file_name = title[ib] + '.txt'
    print file_name
    
    source = requests.get(url)
    print source
    soup = BeautifulSoup(source.content, "lxml")    
    
    try :
        text_file_url = soup.find('a', {'type': 'text/plain'})['href']
        text_file_url = 'https:' + text_file_url
        print text_file_url
        
        try :
            wget.download(text_file_url, file_name)
            print (file_name, "successfully downloaded\n")
            print '-'*60

        except HTTPError as e:
            print ("HTTP error({0}): {1}".format(e.errno, e.strerror))
            error_indx.append(str(ib) + pipe + pg_num[ib] + pipe + title[ib])

    except :
        try :
            text_file_url = soup.find('a', {'charset': 'utf-8'})['href']
            text_file_url = 'https:' + text_file_url
            print text_file_url

            try :
                wget.download(text_file_url, file_name)
                print (file_name, "successfully downloaded\n")
                print '-'*60

            except HTTPError as e:
                print ("HTTP error({0}): {1}".format(e.errno, e.strerror))
                error_indx.append(str(ib) + pipe + pg_num[ib] + pipe + title[ib])

        except :
            print ("Text link not found\n")
            print '~'*60

print('... file downloads complete.')

error_indx

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... end_of_file
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

