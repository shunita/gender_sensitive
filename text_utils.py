import nltk
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet
import re
import string

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN
        # return ''

def is_english_word(word_to_check):
    return len(wordnet.synsets(word_to_check)) > 0


def represents_number(word_to_check):
    possible_numbers = re.findall(r'[\d\.]+', word_to_check)
    if len(possible_numbers) > 0 and len(possible_numbers[0]) == len(word_to_check):
        return True
    syns = wordnet.synsets(word_to_check)
    for s in syns:
        if s.definition().startswith("the cardinal number"):
            return True
    if "-" in word_to_check:
        word = word_to_check.split("-")[0]
        syns = wordnet.synsets(word)
        for s in syns:
            if s.definition().startswith("the cardinal number"):
                return True
    return False


def stem_carefully(word):
    stemmer = SnowballStemmer('english')
    stemmed = stemmer.stem(word)
    if is_english_word(stemmed):
        return stemmed
    else:
        return word


def clean(text):
    """tokenize, remove stop words and numbers, and lemmatize."""
    stop_words = stopwords.words('english')

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos = pos_tag(tokens)
    lemm = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(p)).lower() for word, p in pos]
    without_stop_words = [word for word in lemm
                          if word not in stop_words
                          and word not in string.punctuation
                          and not represents_number(word)
                          and is_english_word(word)]
    return without_stop_words


def decide_on_word(word):
    w = word.lower().strip(string.punctuation)
    if represents_number(w):
        return "<NUMBER>"
    return w


def break_and_lemmatize(text):
    """tokenize, remove stop words and numbers, and lemmatize."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    # some words have a slash that is not caught by word_tokenize
    tokens1 = []
    for token in tokens:
        s = [t for t in token.strip(string.punctuation).split("/") if t != ""]
        tokens1.extend(s)
    pos = pos_tag(tokens1)
    lemm = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(p)).lower() for word, p in pos]
    without = [decide_on_word(word) for word in lemm if word not in string.punctuation]
    return without

def flatten_list_of_list_of_texts(lst):
    return [item.strip() for sublist in lst for item in sublist]

def sentence_break(text):
    return flatten_list_of_list_of_texts([sent_tokenize(x) for x in text.split(";")])


def split_abstract_to_sentences(abstract):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    parts = abstract.split(';')
    sentences = []
    for part in parts:
        sentences.extend([s.strip('[]()') for s in sent_tokenizer.tokenize(part)])
    return sentences
