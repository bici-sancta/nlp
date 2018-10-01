
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

