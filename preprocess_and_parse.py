
# coding: utf-8

# # StanfordCoreNLP parser
# ##### Currently using the parser only, add NER later
# git - https://github.com/smilli/py-corenlp
# <br>
# how to run local web server - https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started
# <br>
# on output formats - https://stanfordnlp.github.io/CoreNLP/corenlp-server.html
# <br><br>
# Run in cmd to start server
# <br>
# cd C:\stanford-corenlp-full-2017-06-09
# <br>
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# 

# API server setup
from pycorenlp import StanfordCoreNLP
import re

# Initiate CorNLP object
nlp = StanfordCoreNLP('http://localhost:9000')



import re

def preprocess(file_dir):
    """
    Removes special chars , title
    Normalizes spaces>2  to one
    """

    title = re.compile(r"%&%.*%&%")
    special_chars = re.compile(r"[!@##$$%^&*(),:\"]") 
    parag_tag = re.compile("<p>")
    
    text = open(file_dir, 'r', encoding = 'utf-8').read()
    
    text = re.sub(title, "", text)
    text = re.sub(special_chars, "", text)
    text = re.sub(parag_tag, "", text)
    text = re.sub("\s{2,}", " ",text) # Normalize 2 > whitespace to 1 whitespace
    
    return text



from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag): 
    """
    Changes treebank tags to Wordnet tags to be fed into the WordNet lemmatizer
    Returns -1 except for ADJ, VERB, NOUN, ADV
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
    else:
        return -1



from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 

def parse_text(text, lemmatize = False,
                     stopWords_filter = False,
                     show_pos = True):
    """
    Parses text via the Stanford parser, filters outuput based on POS tags to be fed into lemmatizer   
    Removes stop words from the output
    
    Params
    lemmatize: uses WordNet to normalized words. If pos_filter ins enabled, POS is fed into the lemmatizer too. 
    For a list of POS recognized by the WordNet lemmatizer, refer to get_wordnet_pos
    stopWords_filter: removes words that are in the stop-word list
    show_pos = returns POS tags with words as a pair
    
    API parameters for Standford parser
    annotators: tokenize, ssplit, pos, lemma, ner, parse, dcoref
    outputFormats: text, json, xml, Serialized
    """
    stopWords = []
    if (stopWords_filter == True):
        stopWords = stopwords.words('English')
        with open(r"C:\nlp\extra_stopwords.txt", 'r', encoding = 'UTF-8') as f:
            extra_stopWords = f.read()
            extra_stopWords = extra_stopWords.split("\n")
            stopWords.append(extra_stopWords)
            
    if (lemmatize == True):
        wordNet = WordNetLemmatizer()

        
    text = text.lower()
    
    output = nlp.annotate(text, properties={
            'annotators': 'ssplit, pos', 
            'outputFormat': 'json'
            })
    
    word_tags_list = []
    for sentence in output['sentences']:
        for item in sentence['tokens']:

            word = item['word']
            pos = item['pos']

            # Do not add to list if:
            if (lemmatize == True) & (get_wordnet_pos(pos) == -1): # kept POS: ADJ, VERB, NOUN, ADV
                continue
            if (stopWords_filter == True) & (word in stopWords):
                continue
            
            if (lemmatize == True):
                word = wordNet.lemmatize(word)
            
            # Append to list
            if (show_pos == False):
                word_tags_list.append(word)
            elif (show_pos == True):
                word_tags_list.append(word + "_"+ pos)
            
    return word_tags_list
