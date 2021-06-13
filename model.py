
import json
import re
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import collections
import tensorflow as tf
import numpy as np
import random

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TURNING ON GPU ACCELERATION

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# LOADING DATA

all_records = []
with open('news_dataset.json') as json_file:
    for line in json_file:
        record = json.loads(line)
        all_records.append(record)

# PREPROCESSING DATA

for record in all_records:
    record['snippet'] = record['snippet'].replace("&#39;", "'")
    record['snippet'] = text_to_word_sequence(record['snippet'])

# CATEGORIZING DATA

snippets = [record['snippet'] for record in all_records]
topics = [record['topic'] for record in all_records]
labels = [1 if record['topic'] == 'Football' else 0 for record in all_records]

football_related = [(record['snippet'], 1) for record in all_records if record['topic'] == 'Football']
non_football_related = [(record['snippet'], 0) for record in all_records if record['topic'] != 'Football']

random.shuffle(football_related)
random.shuffle(non_football_related)

num_of_football_snippets_training = 20  # 31 max
num_of_other_snippets_training = 20  # 571 max
num_of_football_snippets_testing = len(football_related) - num_of_football_snippets_training
num_of_other_snippets_testing = len(non_football_related) - num_of_other_snippets_training

training_dataset = football_related[:num_of_football_snippets_training] + non_football_related[
                                                                          :num_of_other_snippets_training]
random.shuffle(training_dataset)

test_dataset = football_related[
               num_of_football_snippets_training:num_of_football_snippets_training + num_of_football_snippets_testing] + non_football_related[
                                                                                                                         num_of_other_snippets_training:num_of_other_snippets_testing + num_of_other_snippets_training]
random.shuffle(test_dataset)

training_data = [datapoint[0] for datapoint in training_dataset]
training_labels = [datapoint[1] for datapoint in training_dataset]

test_data = [datapoint[0] for datapoint in test_dataset]
test_labels = [datapoint[1] for datapoint in test_dataset]

# TOKENIZING WORDS (it's important to fit tokenizer on training set only)

oov_token = "<OOV>"
max_word_dictionary_size = 5000  # around 4000 unique words in whole dataset

tokenizer = Tokenizer(num_words=max_word_dictionary_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_data)

training_sequences = tokenizer.texts_to_sequences(training_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

# PADDING WORDS (filling missing space in vectors with 0)

padding_type = 'pre'  # more commonly used practice
max_seq_len = 100
training_padded = pad_sequences(training_sequences, maxlen=max_seq_len, padding=padding_type)
test_padded = pad_sequences(test_sequences, maxlen=max_seq_len, padding=padding_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(test_padded)
testing_labels = np.array(test_labels)

# LOADING GLOVE WORD EMBEDDINGS

path_to_glove_file = 'glove.6B.300d.txt'
embeddings_index = {}

with open(path_to_glove_file, mode='r', encoding="utf-8") as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

VOCAB_SIZE = len(word_index) + 1

# CREATING MODEL

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], input_length=max_seq_len,
                              trainable=False),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
epochs = 200
model.fit(x=training_padded, y=training_labels, epochs=epochs, validation_data=(test_padded, testing_labels))

# EVALUATING EXACT STATS

dataset = football_related + non_football_related
random.shuffle(dataset)
dataset_snippets = [datapoint[0] for datapoint in dataset]
dataset_labels = [datapoint[1] for datapoint in dataset]
positive_positives = 0
positive_negatives = 0
negative_positives = 0
negative_negatives = 0

sequences = tokenizer.texts_to_sequences(dataset_snippets)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding=padding_type)
padded_sequences = np.array(padded_sequences)
labels = np.array(dataset_labels)

for snippet, label in zip(padded_sequences, labels):
    snippet = np.array(snippet).reshape(1, snippet.shape[0])
    predicition = model.predict([snippet])[0][0]
    if label == 1:
        if predicition >= 0.5:
            positive_positives += 1
        else:
            positive_negatives += 1
    else:
        if predicition <= 0.5:
            negative_negatives += 1
        else:
            negative_positives += 1

print(
    f'model classified correctly {positive_positives} out of {len(football_related)} football snippets and falsely assigned {positive_negatives} out of {len(non_football_related)} as football related while they were not')
