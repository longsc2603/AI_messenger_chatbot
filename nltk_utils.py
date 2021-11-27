import nltk
nltk.download('punkt')  
from nltk.stem.porter import PorterStemmer
import numpy as np


stemmer = PorterStemmer()
punctuation = ['?', '.', ',', '!', ':', '/']


def token(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = sorted(set([stem(w) for w in tokenized_sentence
                                     if w not in punctuation]))
    bag = np.zeros(len(all_words), dtype=np.float32)
    for (id, w) in enumerate(all_words):
        if w in tokenized_sentence:
            bag[id] = 1
    return bag
