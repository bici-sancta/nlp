
# coding: utf-8

# In[7]:


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... ref : https://github.com/dipanjanS/text-analytics-with-python
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import import_ipynb # supports importing other ipynb notebooks

from contractions import CONTRACTION_MAP

import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
from HTMLParser import HTMLParser
import unicodedata

stopword_list = nltk.corpus.stopwords.words('english')

# ... add stopwords that are common for film reviews

stopword_list = stopword_list + ['film', 'movie', 'watch', 'cinema', 'scene',
                                 'story', 'show']
stopword_list = stopword_list + ['mr', 'mrs', 'come', 'go', 'get',
                                 'tell', 'listen', 'one', 'two', 'three',
                                 'four', 'five', 'six', 'seven', 'eight',
                                 'nine', 'zero', 'join', 'find', 'make',
                                 'say', 'ask', 'tell', 'see', 'try', 'back',
                                 'also']


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... add stopwords for novel topic modeling - common names (5,000+)
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
stopword_list = stopword_list + ['ii', 'iii', 'iv', 'v', 'vi',
                                 'vii', 'viii', 'ix', 'x', 'xi',
                                 'xii', 'xiii', 'xiv', 'xv', 'xvi',
                                 'xvii', 'xviii', 'xix', 'xx']

text_file = open("more_stop_words.csv", "r")
lines = text_file.read().split(',')
clean_list = [re.sub(' ', '', x) for x in lines]
clean_list

stopword_list = stopword_list + clean_list

text_file = open("dickens_character_names.csv", "r")
lines = text_file.read().split(',')
clean_list = [re.sub(' ', '', x) for x in lines]
clean_list = [x.lower() for x in clean_list]
clean_list = list(set(clean_list))

stopword_list = stopword_list + clean_list

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

wnl = WordNetLemmatizer()
html_parser = HTMLParser()

def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

def expand_contractions(text, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)                                if contraction_mapping.get(match)                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
    
    
from pattern.en import tag
from nltk.corpus import wordnet as wn

# Annotate text tokens with POS tags
def pos_tag_text(text):
    
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    tagged_text = tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text
    
# lemmatize text based on POS tags 
def lemmatize_text(text):
    
    pos_tagged_text = pos_tag_text(text)
    
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word                     
                         for word, pos_tag in pos_tagged_text]
    
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text
    

def remove_special_characters(text):
    
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text
    
    
def remove_stopwords(text):
    
    tokens = tokenize_text(text)
    
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    
    return filtered_text

def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def unescape_html(parser, text):
    
    return parser.unescape(text)

def normalize_corpus(corpus, lemmatize = True, 
                     only_text_chars = False,
                     tokenize = False):
    
    normalized_corpus = []
    
    for text in corpus:
        
        text = html_parser.unescape(text)
        text = expand_contractions(text, CONTRACTION_MAP)
        
        if lemmatize:
            text = lemmatize_text(text)
            
        else:
            text = text.lower()
                
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        
        if only_text_chars:
            text = keep_text_characters(text)
        
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)
            
    return normalized_corpus

def call_pos_tagger(text):
    
    pos_tagged_text = pos_tag_text(text)
    
    return pos_tagged_text


def normalize_corpus_NV(corpus, lemmatize = True, 
                     only_text_chars = False,
                     tokenize = False):
    
    normalized_corpus = []
    
    for text in corpus:
        
        text = html_parser.unescape(text)
        text = expand_contractions(text, CONTRACTION_MAP)
        
        if lemmatize:
            text = lemmatize_text(text)
            
        else:
            text = text.lower()
            
# ... need to re-think this ... this is pos tagging after lemmatizing
# ... this is likely not getting best POS results

        pos_tagged_text = call_pos_tagger(text)
        
        text = [word if (pos_tag == 'n' or pos_tag == 'v') else ';'
                    for word, pos_tag in pos_tagged_text]
        
        text = ' '.join(text)
        
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        
        if only_text_chars:
            text = keep_text_characters(text)
        
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)
            
    return normalized_corpus

def parse_document(document):
    document = re.sub('\n', ' ', document)
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize('NFKD', document).encode('ascii', 'ignore')
    else:
        raise ValueError('Document is not string or unicode!')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences
