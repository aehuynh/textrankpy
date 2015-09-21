from utils.text_process import clean_sentence, lemmatize_sentence

from nltk import sent_tokenize
from networkx import Graph, pagerank

from math import log10, floor
from itertools import combinations
from operator import attrgetter

# EXTRACTION_RATIO: The ratio of the total number of sentences to extract
EXTRACTION_RATIO = 0.33

class Sentence(object):
    """An object representing a sentence. It encapsulates all the metadata
    of a sentence used for TextRank.

    :param text: The original sentence string
    :param text_place: The position of this sentence in the text it came from
    :param reduced: The reduced/cleaned form of the sentence
    """
    def __init__(self, text, text_place, reduced=None):
        self.text = text
        self.text_place = text_place
        self.reduced = lemmatize_sentence(clean_sentence(text)) if reduced is None else reduced

def summarize(text, ratio=EXTRACTION_RATIO):
    # Turn text into a list of Sentence objects in the order that they appear
    # in the original text
    sentences = create_sentences(text)

    # Create graph and perform pagerank
    graph = create_graph(sentences)
    pagerank_results = pagerank(graph,weight='weight')

    # Sort the pagerank results and grab only the value of the nodes
    pagerank_results = sorted(pagerank_results, key=pagerank_results.get, reverse=True)

     # Get results in the top (ratio * 100) %
    results_to_extract = floor(len(pagerank_results) * ratio)
    top_results = pagerank_results[0:results_to_extract]

    # Convert the PageRank results into a list of sentences
    # Sort the sentences by the order they appear in the text
    top_text_order_sentences = [sentences[text_place].text for text_place in sorted(top_results)]

    return ' '.join(top_text_order_sentences)

def create_sentences(text):
    """Convert text into a list of Sentence objects.
    """
    sentences = []
    text_place = 0
    for sentence in sent_tokenize(text):
        sentences.append(Sentence(text=sentence, text_place=text_place))
        text_place +=1
    return sentences

def create_graph(sentences):
    """Creates a complete NetworkX graph with the text_place of every sentence
    as the nodes. The edge weights are values indicating the similarity
    between each sentence.
    """
    graph = Graph()

    for sentence in sentences:
        graph.add_node(sentence.text_place)

    # Turn graph into a complete graph
    for (s1, s2) in combinations(sentences, 2):
        # The weight is derived from the textrank sentence similarity equation
        # applied to the reduced forms of the sentences.
        similarity_value = similarity(s1.reduced,s2.reduced)
        graph.add_edge(s1.text_place, s2.text_place, weight=similarity_value)

    return graph

def similarity(s1, s2):
    """Calculates a term frequency based similarity score between two
    sentences. It is based on the equation from the article
    'TextRank: Bringing Orders'.
    """
    words_one = s1.split()
    words_two = s2.split()

    common_word_count = len(set(words_one) & set(words_two))

    log_sum = log10(len(words_one)) + log10(len(words_two))
    if log_sum== 0:
        return 0

    return common_word_count / log_sum
