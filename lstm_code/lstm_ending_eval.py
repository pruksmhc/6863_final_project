
# coding: utf-8

# In[1]:

from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import random
from random import shuffle
import datetime
import csv
import keras
import os
import argparse

parser = argparse.ArgumentParser(description='Pluralizing words')
parser.add_argument('--n', default=250, type=int,
                    help='Number of neurons for the model (choose a multiple of 25 between 25 and 250)')
parser.add_argument('--word', type=str,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
# In[2]:

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# # Data Loading

# In[26]:

batch_size = 64
epochs = 400  
latent_dim = 10
num_samples = 45133

# This flag should be set to 1 for the classification style training
# Classification based task is when the network is expected to output
# 's' for the input 'apple' and 'es' for 'torch'.
# When the classfication flag is set to 0 then the network is expected to 
# output 'apples' for input 'apple' and 'torches' for input 'torch'. 
classification_flag = 1

# Path to the data txt file on disk.
data_path = './all_data.csv'


# In[27]:

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


# In[28]:

train_input = []
train_output = []
train_file = 'train.csv'
if classification_flag:
    train_file = 'train_ending.csv'

with open(train_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        train_input.append(row[0])
        train_output.append(row[1]) # replace with row[1][1:-1] if you don't want \t and \n around the word
        
        


# In[29]:

test_input = []
test_output = []
test_file = 'test.csv'
if classification_flag:
    test_file = 'test_ending.csv'
with open(test_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        test_input.append(row[0])
        test_output.append(row[1]) # replace with row[1][1:-1] if you don't want \t and \n around the word


# In[30]:

input_texts = train_input + test_input
target_texts = train_output + test_output


# In[31]:

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])



# In[43]:

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
input_texts = [args.word]
targert_text = ['']
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


# In[44]:

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# In[45]:

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]

    return decoded_sentence, states_value


# In[46]:

latent_dim = args.n
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))


decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
if classification_flag:
    model.load_weights('./final_ending_models/rmsprop_model_500_' + str(latent_dim) + '.h5')
else:
    model.load_weights('./final_models/rmsprop_model_500_' + str(latent_dim) + '.h5')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())
delta = 0
correct = 0
print('')
for seq_index in range(len(input_texts)):
    input_seq = encoder_input_data[seq_index: seq_index+ 1]
    decoded_sentence, _ = decode_sequence(input_seq)
    print("Input word: ", args.word, "\nPlural word prediction: \'" +  decoded_sentence[:-1] + '\'')


