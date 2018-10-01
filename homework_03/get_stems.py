

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(phrase) :
    
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(phrase)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    return(filtered_sentence)


def get_stems(phrase, method) :
    
    prtr = nltk.stem.PorterStemmer()
    snob = nltk.stem.SnowballStemmer('english')
    lema = nltk.wordnet.WordNetLemmatizer()
    
    words_to_stem = remove_stopwords(phrase)

    stems = [w for w in words_to_stem]
    stems = []
    
    if method == 'porter' :
        for w in words_to_stem:
            stems.append(prtr.stem(w))
 
    elif method == 'snowball': 
        for w in words_to_stem:
            stems.append(snob.stem(w))
            
    elif method == 'lemmatize': 
        for w in words_to_stem:
            stems.append(lema.lemmatize(w))

    return (stems)

