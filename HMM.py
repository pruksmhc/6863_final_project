from seqlearn.hmm import MultinomialHMM
import pandas as pd
import string
import numpy as np

#helper function for evaluation
def evaluate(prediction, target):
	return sum(1 for i,j in zip(prediction,target) if i == j)*1.0/len(prediction)


#import data from csv file
model = MultinomialHMM()
data = pd.read_csv("all_data.csv")
alphabet_dict = dict(zip(string.ascii_lowercase, range(1,27)))

w_class = 0
X = []
Y = []

for index, row in data.iterrows():
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
	X.append(word_int)
	Y.append(w_class)

#reshaping and padding array with zeros
padded_X = np.zeros([len(X), len(max(X, key = lambda x: len(x)))])
for i, j in enumerate(X):
	padded_X[i][0:len(j)] = j

print "TOTAL NUMBER OF SAMPLES: ", len(padded_X)
#separate into train and validate
train_X = padded_X[:35000]
train_Y = Y[:35000]
val_X = padded_X[35000:]
val_Y = Y[35000:]

#fit the model to training data
model.fit(train_X, train_Y, 10)

#evaluation
#problem I noticed: predicting everthing as 1 due to imbalance of data
print "ACCURACY: ", evaluate(model.predict(val_X), val_Y)



