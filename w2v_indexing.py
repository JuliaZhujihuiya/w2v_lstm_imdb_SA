import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import os
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import Word2Vec


def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review,  'lxml').get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        words = np.array(words)
    # 5. Return an array of words
    return words


def create_dictionary(txt_path):
    word_list = []
    vocabulary_vectors = []
    with open(txt_path, encoding="utf-8") as data:
        next(data)
        for line in data:
            temp = line.strip().split(" ")
            word = temp[0]
            word_list.append(word.lower())
            vector = [temp[i] for i in range(1, len(temp))]
            vector = list(map(float, vector))
            vocabulary_vectors.append(vector)
        # saving data
        vocabulary_vectors = np.array(vocabulary_vectors) # vectors
        word_list = np.array(word_list) # words
        np.save(r"process/vocabulary_vectors", vocabulary_vectors)
        np.save(r"process/word_list", word_list)
        return vocabulary_vectors, word_list


# processing labeled training data
# format: res[0] = [[list of words] , sentiment]]
def process_train(data):
    res = []
    print("processing labeled training data to lstm training pair")
    for i in range(len(data["review"])):
        if (1+1)%1000 == 0:
            print("review", i, "of 25000")
        txt = data["review"][i]
        txt = review_to_wordlist(txt, remove_stopwords = True)
        print(txt.dtype)
        sent = np.array(data["sentiment"][i])
        print(sent.dtype)
        temp = [txt, sent]
        res.append(temp)
    np.save(r"process/lstm_train_data", res)
    return res

def process_test(data):
    res = []
    print("processing test data...")
    for i in range(len(data["review"])):
        if (1+1)%1000 == 0:
            print("review", i, "of 25000")
        txt = data["review"][i]
        txt = review_to_wordlist(txt, remove_stopwords = True)
        # print(txt.dtype)
        res.append(txt)
    np.save(r"process/lstm_test_data", res)
    return res

def process_sentence(data):
    print("processing sentence to vectors...")
    # data = lstm_train_data
    sentence_code = np.zeros((25000, 200), dtype=np.int32)

    word_list = np.load(r"process/word_list.npy", allow_pickle=True)
    word_list = word_list.tolist()

    for i in range(len(data)):
        if (i+1) % 1000 == 0:
            print("sentence", i)
        words = data[i]
        for j in range(len(words)):
            word = words[j]
            try:
                index = word_list.index(word) + 1
            except ValueError:
                index = 0
            if j >= 200:
                break
            else:
                sentence_code[i][j] = index
    print(sentence_code, sentence_code.dtype)
    print(sentence_code[0], sentence_code[0].dtype)
    np.save(r"process/sentence_code_test", sentence_code)
    return sentence_code

if __name__ == '__main__':
    # Read data
    # train = pd.read_csv(r"Data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    # test = pd.read_csv(r"Data/testData.tsv", header=0, delimiter="\t", quoting=3)
    # unlabeled_train = pd.read_csv(r"Data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    # load w2v_model
    w2v_model = Word2Vec.load(r"process/300features_40minwords_10context")
    # w2v_model.wv.save_word2vec_format(r"process/w2v_300d.txt", binary=False)
    # w2v_model = w2v_model.wv.load_word2vec_format(r"process/w2v_300d.txt")
    # create_dictionary(r"process/w2v_300d.txt")
    print(w2v_model.wv['review'])
    # process_train(train)
    # process_test(test)

    # lstm_train_data = np.load(r"process/lstm_train_data.npy", allow_pickle = True)
    # process_sentence(lstm_train_data)
    # lstm_test_data = np.load(r"process/lstm_test_data.npy", allow_pickle = True)
    # process_sentence(lstm_test_data)