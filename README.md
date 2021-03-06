# TextrankPy
A keyword extractor and text summarizer

### Table of Contents

1. [Description](#description)

2. [PageRank](#pagerank)

3. [Lemmatization/Stemming](#lemmatizationstemming)

4. [Co-occurence Window](#co-occurence-window)

5. [Example Summary](#example-summary)

###  Description
Both the keyword extractor and text summarizer are based on the TextRank algorithm described in <a href="http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf">TextRank: Bringing Order into Texts<a>. 
The text summarizer extracts the most important sentences of a text. 

###  PageRank
This project includes a PageRank implementation done mainly for fun. The underlying graph structure used (NetworkX graph) 
includes a native implementation of PageRank.

###  Lemmatization/Stemming
There were some ambiguities about the exact implementation of TextRank, especially on the role of word reduction.
In TextrankPy:

1. It lemmatizes/stems words before adding them to the graph and creating the edges
2. After PageRank is run on the reduced forms of the words, all the original words in the text with the top lemmas/stems
are considered as top keywords

###  Co-occurence Window
TextrankPy keeps the co-occurence window within a sentence. It will not create edges for words that are technically
within the same co-occurence window but are in different sentences.

###  Example Summary
A summary of TextRank(<a href="https://en.wikipedia.org/wiki/Automatic_summarization#Unsupervised_approach:_TextRank">Unsupervised approach: TextRank<a>) generated by TextrankPy:

>Instead of trying to learn explicit features that characterize keyphrases, the TextRank algorithm exploits the structure of the text itself to determine keyphrases that appear "central" to the text in the same way that PageRank selects important Web pages. In this way, TextRank does not rely on any previous training data at all, but rather can be run on any arbitrary piece of text, and it can produce output simply based on the text's intrinsic properties. For keyphrase extraction, it builds a graph using some set of text units as vertices. Edges are based on some measure of semantic or lexical similarity between the text unit vertices. However, to keep the graph small, the authors decide to rank individual unigrams in a first step, and then include a second step that merges highly ranked adjacent unigrams to form multi-word phrases. For example, if we rank unigrams and find that "advanced", "natural", "language", and "processing" all get high ranks, then we would look at the original text and see that these words appear consecutively and create a final keyphrase using all four together. Two vertices are connected by an edge if the unigrams appear within a window of size N in the original text. These edges build on the notion of "text cohesion" and the idea that words that appear near each other are likely related in a meaningful way and "recommend" each other to the reader. Since this method simply ranks the individual vertices, we need a way to threshold or produce a limited number of keyphrases. It is not initially clear why applying PageRank to a co-occurrence graph would produce useful keyphrases. Similarly, if the text contains the phrase "supervised classification", then there would be an edge between "supervised" and "classification". In the final post-processing step, we would then end up with keyphrases "supervised learning" and "supervised classification".
