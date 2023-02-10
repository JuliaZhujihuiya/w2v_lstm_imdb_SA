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

from lstm_train import LSTM


def test(batch_num=250, batch_size=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('loading data...')
    test_data = pd.read_csv(r"Data/testData.tsv", header=0, delimiter="\t", quoting=3)

    model = LSTM(hidden_size=64).to(device)
    model.load_state_dict(torch.load('lstm_model/LSTM_epoch25.pkl')['model'])
    model.eval()

    res = []
    for batch in range(batch_num):
        arr_test_add = r"test/arr_test/arr_test_" + str(batch) + ".npy"
        arr_test = np.load(arr_test_add, allow_pickle=True)
        print(f'testing batch {batch}...')

        for i in range(batch_size):
            input_ = arr_test[i]
            input_ = torch.tensor(input_, dtype=torch.float32).to(device)
            output = model(input_)
            pred = output.max(dim=-1)[1]
            pred = pred.item()
            res.append(pred)

    output = pd.DataFrame(data={"id": test_data["id"], "sentiment": res})
    # Use pandas to write the comma-separated output file
    output.to_csv("test/lstm_model_epoch25.csv", index=False, quoting=3)


if __name__ == '__main__':
    test()
