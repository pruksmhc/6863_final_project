from seqlearn.hmm import MultinomialHMM
import pandas as pd
import string
from hmmlearn import hmm
import numpy as np
"""
Trying out HMMLearn
https://github.com/hmmlearn/hmmlearn

"""
all_data = pd.read_csv("all_data.csv")
# encode
alphabet_dict = dict(zip(string.ascii_lowercase, range(1,27)))
lengths = []
X = pd.DataFrame()
for index, row in all_data.iterrows():
    plural = row[1]
    word_int = []
    for let in plural:
        word_int.append(alphabet_dict[let])
    X = X.append(word_int)
    lengths.append(len(word_int))

print("now fit")
model = hmm.GaussianHMM(n_components=3)
model.fit(X, lengths)
test = "apple"
test_int = []
for let in test:
    test_int.append(alphabet_dict[let])
pdb.set_trace()
print(model.predict(np.array([test_int])))
print(alphabet_dict)

'''
Documentation of seqlearn: http://larsmans.github.io/seqlearn/reference.html
Some examples of HMMs using seqlearn
seqlearn is supervised leanring vs hmmlearn, which is unsupervised.

'''

model = MultinomialHMM()
# encode

X = pd.DataFrame()
y = pd.DataFrame()
# Here 0 represents pluarl = singuar + '', 1 represents plural = singular +s, and 2 represnets plural = singular + 'es'
# data preparation
# index the words such that a = 1, z = 26.
w_class = 0
import pdb;
pdb.set_trace()
for index, row in all_data.iterrows():
    singular = row[0]
    plural = row[1]
    if (plural[-2] == 'es'):
        w_class = 2
    elif (plural[-1] == 's'):
        w_class = 1
    else:
        w_class = 0
    word_int = []
    for let in singular:
        word_int.append(alphabet_dict[let])
    X = X.append(word_int)
    y = y.append([w_class])
pdb.set_trace()
length = 10
# fit the HMM
model.fit(X, y, length)
