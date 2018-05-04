from seqlearn.hmm import MultinomialHMM
import pandas as pd
import string
import numpy as np

#helper function for evaluation
def evaluate_with_output(prediction, target, input_X):
	# prediction + target
	print("ACCURACY" + str(sum(1 for i,j in zip(prediction,target) if i == j)*1.0/len(prediction)))
	for index in range(len(input_X)):
		n_word = ''
		#import pdb; pdb.set_trace()
		for i in input_X[int(index)]:
			if (i != 0):
				n_word += reverse_dict[int(i)]
		print("singular " + n_word)
		if (prediction[int(index)] == 0):
			print(n_word)
		elif (prediction[int(index)] == 1):
			print(n_word +"s")
		elif (prediction[int(index)] == 3):
			print(n_word[:-1] + "ies")
		else:
			print(n_word +"es")
#just return accuracy for tuning on dev set
def evaluate(prediction, target, input_x):
	return sum(1 for i,j in zip(prediction,target) if i == j)*1.0/len(prediction)


#import data from csv file
model = MultinomialHMM()
data = pd.read_csv("weighted_data.csv")


alphabet_dict = dict(zip(string.ascii_lowercase, range(1,27)))
reverse_dict = dict(zip(range(1,27), string.ascii_lowercase))
X = []
Y = []

for index, row in data.iterrows():
	w_class = 4
	singular = row[0]
	plural = row[1]
	singular_without_end = singular[:len(singular)-1]
	#append -es cases
	if (plural == singular + 'es'):
		w_class = 2
	#simply append -s cases
	elif (plural == singular + 's'): 
		w_class = 1
	#endings with -y
	elif ((len(singular) > 1) and (plural == singular_without_end + 'ies')): # y -> ies
		w_class = 3
	elif (plural == singular): # don't add anything.
		w_class = 0
	if (w_class != 4):
		word_int = []
		plural_int = []
		for let in singular:
			word_int.append(alphabet_dict[let])
		X.append(word_int)
		Y.append(w_class)

#reshaping and padding array with zeros
# get the max length of input
padded_X = np.zeros([len(X), len(max(X, key = lambda x: len(x)))])
for i, j in enumerate(X):
	padded_X[i][0:len(j)] = j


print "TOTAL NUMBER OF SAMPLES: ", len(padded_X)
#separate into train and validate
train_X = padded_X[:10000]
train_Y = Y[:10000]
val_X = padded_X[10000:]
val_Y = Y[10000:]

#fit the model to training data
# model.fit(train_X, train_Y, 10)
# print evaluate(model.predict(val_X), val_Y, val_X)

# tuning on the dev set for the optimal number of hidden states
best_accuracy = 0
best_number = 1
for n in range(50):
	model.fit(train_X, train_Y, n)
	accuracy = evaluate(model.predict(val_X), val_Y, val_X)
	print "n = ", n
	print accuracy
	if accuracy > best_accuracy:
		best_accuracy = accuracy
		best_number = n

#evaluation on best number of hidden states
model.fit(train_X, train_Y, best_number)
#best accuracy: 0.48
print "Accuray: " , evaluate(model.predict(val_X), val_Y, val_X)
print evaluate_with_output(model.predict(val_X), val_Y, val_X)
