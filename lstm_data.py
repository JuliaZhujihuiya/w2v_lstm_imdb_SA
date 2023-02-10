import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import os
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import Word2Vec

# Read data
# train = pd.read_csv(r"Data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
# test = pd.read_csv(r"Data/testData.tsv", header=0, delimiter="\t", quoting=3)
# unlabeled_train = pd.read_csv(r"Data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)


def process_batch(batch_size = 100):
    #sentences = 25000
    #every batch 250 sentences

    lstm_train_data = np.load(r"process/lstm_train_data.npy", allow_pickle=True)
    sentence_code = np.load(r"process/sentence_code.npy", allow_pickle=True)
    vocabulary_vectors = np.load(r"process/vocabulary_vectors.npy", allow_pickle=True)

    batch_num = int(25000 / batch_size) # batch_num = 250

    # print("index to vectors...")
    # sentence_code = sentence_code.reshape((batch_num, batch_size, 200))
    # for batch in range(batch_num):
    #     print("batch", batch+1, "of", batch_num)
    #     arr_train = np.zeros((batch_size, 200, 300))
    #     for i in range(batch_size):
    #         for j in range(200):
    #             if sentence_code[batch][i][j] != 0:
    #                 index = sentence_code[batch][i][j]-1
    #                 if index == -1:
    #                     break
    #                 arr_train[i][j][:] = vocabulary_vectors[index][:]
    #     batch_name = r"process/arr_train/arr_train_" + str(batch)
    #     np.save(batch_name, arr_train)

    print("batching labels")
    lstm_train_data = lstm_train_data.reshape((batch_num, batch_size, 2))
    for batch in range(batch_num):
        print("batch", batch+1, "of", batch_num)
        labels_train = np.zeros(batch_size, dtype='int8')
        for i in range(batch_size):
            labels_train[i] = lstm_train_data[batch][i][1]
        batch_name = r"process/labels_train/labels_train_" + str(batch)
        np.save(batch_name, labels_train)

def process_test_batch(batch_size = 100):
    #sentences = 25000
    #every batch 250 sentences

    lstm_test_data = np.load(r"process/lstm_test_data.npy", allow_pickle=True)
    sentence_code_test = np.load(r"process/sentence_code_test.npy", allow_pickle=True)
    vocabulary_vectors = np.load(r"process/vocabulary_vectors.npy", allow_pickle=True)

    batch_num = int(25000 / batch_size) # batch_num = 250

    print("index to vectors...")
    sentence_code_test = sentence_code_test.reshape((batch_num, batch_size, 200))
    for batch in range(batch_num):
        print("batch", batch+1, "of", batch_num)
        arr_test = np.zeros((batch_size, 200, 300))
        for i in range(batch_size):
            for j in range(200):
                if sentence_code_test[batch][i][j] != 0:
                    index = sentence_code_test[batch][i][j]-1
                    if index == -1:
                        break
                    arr_test[i][j][:] = vocabulary_vectors[index][:]
        batch_name = r"test/arr_test/arr_test_" + str(batch)
        np.save(batch_name, arr_test)




if __name__ == '__main__':
    # process_batch()
    process_test_batch()


