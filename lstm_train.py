import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import os
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import Word2Vec
import torch
from torch import nn, optim, device


class LSTM(nn.Module):
    def __init__(self, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=hidden_size, num_layers=1,
                            batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(hidden_size, 32),
                                nn.Linear(32, 2),
                                nn.ReLU())

    def forward(self, input_seq):
        x, _ = self.lstm(input_seq)
        x = self.fc(x)
        # x = x[:, -1, :]
        x = x[:][-1][:]

        return x


def train():
    print('loading...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_num = 25
    print('training...')
    model = LSTM(hidden_size=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    criterion = nn.CrossEntropyLoss().to(device)
    loss = 0
    for i in range(epoch_num):
        print(f"start training epoch {i + 1}")
        for j in range(250):
            arr_address = r"process/arr_train/arr_train_" + str(j) + ".npy"
            labels_address = r"process/labels_train/labels_train_" + str(j) + ".npy"
            arr_train = np.load(arr_address, allow_pickle=True)
            labels_train = np.load(labels_address, allow_pickle=True)

            x = arr_train
            # # x = np.atleast_3d(x)
            # temp_x =np.zeros((len(x),len(x[0]),len(x[0][0])))
            # print(temp_x.dtype)
            # print('dim', temp_x.ndim)  # 矩阵的维度
            # print('shape', temp_x.shape)  # 几行几列
            # print('size', temp_x.size)
            # for aa in range(len(x)):
            #     x[aa] = np.array(x[aa])
            #     for bb in range(len(x[aa])):
            #         x[aa][bb] = np.array(x[aa][bb])
            #         for cc in range(50):
            #             # print(x[aa][bb][cc])
            #             x[aa][bb][cc] = np.array(x[aa][bb][cc])
            #             # print(x[aa][bb][cc])
            #             x[aa][bb][cc] = x[aa][bb][cc].astype(float)
            #             temp_x[aa][bb][cc] = x[aa][bb][cc]
            #         x[aa][bb] = x[aa][bb].astype(float)
            #     # x[aa] = x[aa].astype(float)
            # x = np.array(temp_x)
            # x = x.astype(float)
            # print(x.dtype)
            # print('dim', x.ndim)  # 矩阵的维度
            # print('shape', x.shape)  # 几行几列
            # print('size', x.size)
            # print(type(x))
            # x = x.astype(float)  # numpy强制类型转换
            y = labels_train
            # print(y)
            input_ = torch.tensor(x, dtype=torch.float32).to(device)
            label = torch.tensor(y, dtype=torch.long).to(device)
            output = model(input_)
            # print(output)
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print('epoch:%d loss:%.5f' % (i + 1, loss.item()))
    # save model
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, r"lstm_model/LSTM_epoch25.pkl")


if __name__ == '__main__':
    # process_sentence(200)
    # vocabulary_vectors, word_list = create_dictionary(r"process/w2v_300d.txt")
    train()

    # lstm_train_data = process_train(train)
    # data = np.load(r"process/lstm_train_data.npy", allow_pickle=True)
    # vocabulary_vectors = np.load(r"process/vocabulary_vectors.npy", allow_pickle=True)
    # sentence_code = np.load(r"process/sentence_code.npy", allow_pickle=True)
    # process_batch(250)
    # print(sentence_code[0])
    # print(type(sentence_code[0]))
    # print(vocabulary_vectors[1])
    # print(type(vocabulary_vectors[1]))
