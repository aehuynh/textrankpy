from nltk.corpus import stopwords
import string
import re
from nltk.tag import pos_tag
from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()

def clean_sentence(s):
    s = s.lower()
    clean_methods = [remove_punc, remove_numbers, remove_stopwords]
    return clean_sentence_with(s, clean_methods)

def clean_sentence_with(s, clean_methods):
    for method in clean_methods:
        s = method(s)
    return s

def remove_punc(text):
    return text.translate({ord(c): None for c in string.punctuation})

def remove_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])

def remove_stopwords(sentence, lang='english'):
    return ' '.join(filter(lambda x: x.lower() not in stopwords.words(lang), word_tokenize(sentence)))

def lemmatize_sentence(s):
    return ' '.join(map(_lemmatizer.lemmatize, word_tokenize(s)))

def reduce_word(word, use_lemmatizer):
    return _lemmatizer.lemmatize(word) if use_lemmatizer else _stemmer.stem(word)

def filter_by_pos(sentence, pos):
    """Returns the sentence with only words whose part of speech is in
    ACCEPTED_POS.
    """
    words_with_pos = pos_tag(word_tokenize(sentence))
    words_with_pos = filter(lambda word: word[1] in pos, words_with_pos)
    return ' '.join(map(lambda word_with_pos: word_with_pos[0], words_with_pos))
