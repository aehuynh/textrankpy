from nltk.corpus import stopwords
import string
import re

def remove_punc(text):
    return text.translate({ord(c): None for c in string.punctuation})

def remove_stopwords(tokens, lang='english'):
    return list(filter(lambda x: x.lower() not in stopwords.words(lang), tokens))
