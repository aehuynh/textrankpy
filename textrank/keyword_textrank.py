from utils.text_process import clean_sentence, reduce_word, filter_by_pos
from pagerank import PageRankGraph

from nltk import word_tokenize, sent_tokenize
from collections import defaultdict

"""A list of accepted parts of speech for keywords. Only words of these parts
of speech will be used in textrank.

NN = Noun
JJ = Adjective
"""
ACCEPTED_POS = ["NN", "JJ"]

# The number of keywords to extract
RANK_THRESHOLD = 15

# USE_LEMMA: When reducing words, lemmatize words if true else stem words
#
# Words are reduced to their base form before adding to the graph for more
# accurate coocurence edges.
USE_LEMMA = True

def extract_keywords(text, rank_threshold=RANK_THRESHOLD):
    """Extract keywords from text. Returns a list of keywords.

    :param rank_threshold: the number of keywords to extract
    """
    # sentence_word_toks: A 2D list of words where the inner list represents a
    # sentence.
    sentence_word_toks = prepare_for_textrank(text)
    words_of_base = create_words_of_base(unique_words(sentence_word_toks))

    # Create graph and perform pagerank
    graph = create_keyword_graph(sentence_word_toks)
    bases = graph.pagerank()

    # Convert the base forms to the actual words/keywords
    keywords = [word for base in bases for word in words_of_base[base]]

    # Extract only rank_threshold keywords
    number_to_extract = min(len(keywords), rank_threshold)
    keywords = keywords[:number_to_extract]

    return keywords

def prepare_for_textrank(text):
    """Clean and transform text into a 2D list of words

    Steps:
        1. Turn text into list of sentences
        2. Turn text to lower case
        3. Remove punctuation, numbers and stop words
        4. Remove all words that are not nouns or adjectives
        5. Tokenize the sentences
    """
    sentences = sent_tokenize(text)
    sentences = map(clean_sentence, sentences)
    sentences = map(lambda s: filter_by_pos(s, pos=ACCEPTED_POS), sentences)
    sentence_word_toks = list(map(word_tokenize, sentences))

    return sentence_word_toks

def unique_words(sentence_word_toks):
    """Create a list of unique words from all the words in every sentence of
    sentence_word_toks
    """
    return set([word for sentence in sentence_word_toks for word in sentence])

def create_words_of_base(words):
    """Returns a dict mapping a word base form to a list of all the
    Words with that base form
    """
    bases_with_words = defaultdict(list)
    for word in words:
        bases_with_words[reduce_word(word, USE_LEMMA)].append(word)
    return bases_with_words

def create_keyword_graph(sentence_word_toks):
    """Create a page rank graph with the base forms of the words in
    sentence_word_toks.

    The coocurrence window edges only applies to words in the same sentence.
    """
    graph = PageRankGraph()
    for words in sentence_word_toks:
        # Reduce all words to their base form
        bases = map(lambda w: reduce_word(w, USE_LEMMA), words)
        graph.add(bases)
    return graph
