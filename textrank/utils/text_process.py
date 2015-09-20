from nltk.corpus import stopwords
import string
import re
from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def remove_punc(text):
    return text.translate({ord(c): None for c in string.punctuation})

def remove_stopwords(tokens, lang='english'):
    return list(filter(lambda x: x.lower() not in stopwords.words(lang), tokens))

def remove_stopwords_from_sentence(s, lang='english'):
    return ' '.join(filter(lambda x: x.lower() not in stopwords.words(lang), word_tokenize(s)))

def remove_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])

def lemmatize_sentence(s):
    return ' '.join(map(lemmatizer.lemmatize, word_tokenize(s)))

def clean_sentence(s):
    s = s.lower()
    clean_methods = [remove_punc, remove_numbers, lemmatize_sentence, remove_stopwords_from_sentence]
    return clean_sentence_with(s, clean_methods)

def clean_sentence_with(s, clean_methods):
    for method in clean_methods:
        s = method(s)
    return s
