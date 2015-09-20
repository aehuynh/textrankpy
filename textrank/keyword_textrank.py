from utils.text_process import remove_punc, remove_stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from pagerank import PageRankGraph

"""A list of accepted parts of speech for keywords. Only words of these parts
of speech will be used in textrank.

NN = Noun
JJ = Adjective
"""
ACCEPTED_POS = ["NN", "JJ"]

# The number of keywords to extract
RANK_THRESHOLD = 15

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
BASE_FORM_CONVERTER = lemmatizer.lemmatize

def extract_keywords(text, rank_threshold=RANK_THRESHOLD):
    """Extract keywords from text. Returns a list of keywords.

    :param rank_threshold: the number of keywords to extract
    """
    sent_tokens_with_base = prepare_for_textrank(text)
    bases_with_words = create_bases_with_words(unique_words(sent_tokens_with_base))

    # Create graph and perform pagerank
    graph = create_keyword_graph(sent_tokens_with_base)
    bases = graph.pagerank()
    bases = bases[:min(len(bases), rank_threshold)]

    # Convert the base forms to actual words/keywords
    keywords = [word for base in bases for word in bases_with_words[base]]

    return keywords

def prepare_for_textrank(text):
    """Clean and transform text into a 2d array of sentences that contains
    words.

    Steps:
        1. Turn text to lower case
        2. Turn text into list of sentences
        3. Remove punctuation
        4. Word tokenize each sentence in the sentence list
        5. Remove stopwords
        6. Remove all words that are not nouns or adjectives
        7. Turn each word into a dictionary of the original word and its base
           form
    """
    sentences = sent_tokenize(text.lower())
    sentences = map(remove_punc, sentences)
    sent_tokens = map(word_tokenize, sentences)
    sent_tokens = map(remove_stopwords, sent_tokens)
    sent_tokens = map(filter_by_pos, sent_tokens)
    sent_tokens_with_base = list(map(lambda sent_token: add_base_form(sent_token), sent_tokens))

    return sent_tokens_with_base

def filter_by_pos(words):
    """Returns a list of only words whose part of speech is in ACCEPTED_POS
    """
    words_with_pos = pos_tag(words)
    words_with_pos = filter(lambda word: word[1] in ACCEPTED_POS, words_with_pos)
    return list(map(lambda word_with_pos: word_with_pos[0], words_with_pos))

def add_base_form(words):
    """Add the base form of every word in words.

    Returns a list of dictionaries with the original word and the newly added
    base form.

    :param words: list of strings representing a word
    """
    return list(map(lambda word: {"word": word,"base": BASE_FORM_CONVERTER(word)}, words))

def unique_words(sent_tokens):
    """Create a list of unique words from all the words in every sentence of
    sent_tokens
    """
    return list({word['word']: word for words in sent_tokens for word in words}.values())

def create_bases_with_words(words):
    """Returns a dict mapping a word base form to a list of all the
    words with that base form
    """
    bases_with_words = {}
    for word in words:
        if word['base'] in bases_with_words:
            bases_with_words[word['base']].append(word['word'])
        else:
            bases_with_words[word['base']] = [word['word']]
    return bases_with_words

def create_keyword_graph(sent_tokens):
    """Create a page rank graph with the base forms of the words in
    sent_tokens.
    """
    graph = PageRankGraph()
    for words in sent_tokens:
        bases = map(lambda word: word['base'], words)
        graph.add(bases)
    return graph
