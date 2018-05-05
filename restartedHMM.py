import hmmlearn.hmm as hmm
import sklearn
import hmmlearn
import pandas as pd
import string
import numpy as np
import warnings
from random import randint

def decode(encoding):
	reversal_dict = dict(zip(range(1,27), string.ascii_lowercase))
	decoding = ""
	for num in encoding:
		decoding += reversal_dict[num[0]]
	return decoding

#reading in data
data = pd.read_csv("weighted_data.csv")
alphabet_dict = dict(zip(string.ascii_lowercase, range(1,27)))

train_X = []
train_lengths = []
val_X  = []
val_lengths = []
train_count = 0
for index, row in data.iterrows():
	# print "SINGULAR:", row[0]
	# print "PLURAL: ", row[1]
	plural = row[1]
	plural_encoding = []
	for let in plural:
		plural_encoding.append(alphabet_dict[let])

	if len(plural_encoding) != 0:
		if train_count < 10000:
			train_X.append(plural_encoding)
			train_lengths.append(len(plural_encoding))
			train_count += 1
		else:
			val_X.append(plural_encoding)
			val_lengths.append(len(plural_encoding))




NUM_HIDDEN = 10
train_X = np.concatenate([sample for sample in train_X])
train_X = np.atleast_2d(train_X).T
val_X = np.concatenate([sample for sample in val_X])
val_X = np.atleast_2d(val_X).T

#suppress deprecation warnings
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	model = hmm.GaussianHMM(n_components = NUM_HIDDEN)
	model.fit(train_X, train_lengths)

	index = 0
	for length in val_lengths:
		state_sequence =  model.predict(val_X[index: index + length], [length])
		print "WORD: ", decode(val_X[index: index + length])
		print "STATE SEQUENCE: ", state_sequence
		index += length

	print "WORD SAMPLES FROM FITTED MODEL:"
	NUM_WORDS = 10
	for i in range(NUM_WORDS):
		(word, state_sequence) = model.sample(randint(3, 9))
		word_array = word.astype(int)
		if (word_array >= 1).all() and (word_array < 27).all():
			print "WORD: " , decode(word_array)