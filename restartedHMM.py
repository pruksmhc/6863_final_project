import hmmlearn.hmm as hmm
import sklearn
import hmmlearn
import pandas as pd
import string
import numpy as np
import warnings

def get_predicted_plurals(model, val_X, val_lengths):
	predictions = model.predict(val_X, val_lengths)
	samples = model.sample(10) #generate a plural with 10 letters.
	print(samples)
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

import pdb; pdb.set_trace()
NUM_HIDDEN = 30
train_X = np.concatenate([sample for sample in train_X])
train_X = np.atleast_2d(train_X).T # flatten out the plurals.
val_X = np.concatenate([sample for sample in val_X])
val_X = np.atleast_2d(val_X).T

# for which number of hidden states is this the best.
#suppress deprecation warnings
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	model = hmm.GMMHMM(n_components = NUM_HIDDEN)
	model.fit(train_X, train_lengths)
	get_predicted_plurals(model, val_X, val_lengths)
