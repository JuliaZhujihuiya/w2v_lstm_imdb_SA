# # Import the pandas package, then use the "read_csv" function to read
# # the labeled training data
# import pandas as pd
# train = pd.read_csv("Data\labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
# print(train["review"][0])
# print(train.shape)
# print(train.columns.values)

import nltk
nltk.download()  # Download text data sets, including stop words
