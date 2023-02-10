import numpy as np


# train = pd.read_csv(r"Data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
# test = pd.read_csv(r"Data/testData.tsv", header=0, delimiter="\t", quoting=3)
# unlabeled_train = pd.read_csv(r"Data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# lstm_train_data = np.load(r"process/lstm_train_data.npy", allow_pickle=True)
# lstm_test_data = np.load(r"process/lstm_test_data.npy", allow_pickle=True)
# lstm_train_data_0 = lstm_train_data[0]
# lstm_train_data_00 = lstm_train_data[0][0]
# lstm_train_data_01 = lstm_train_data[0][1]
# vocabulary_vectors = np.load(r"process/vocabulary_vectors.npy", allow_pickle=True)
# word_list = np.load(r"process/word_list.npy", allow_pickle=True)
# sentence_code = np.load(r"process/sentence_code.npy", allow_pickle=True)
# sentence_vec = np.load(r"process/sentence_vec.npy", allow_pickle=True)

# arr_train_0 = np.load(r"process/arr_train/arr_train_0.npy", allow_pickle=True)
# labels_train_0 = np.load(r"process/labels_train/labels_train_0.npy", allow_pickle=True)
# arr_test_0 = np.load(r"test/arr_test/arr_test_0.npy", allow_pickle=True)

import torch
print(torch.__version__)
print(1)

