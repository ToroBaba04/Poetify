import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the training data from the different datasets used for our model
# by opening the file then converting the text in each file  into numbers

filepath1 = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
shakespeare_text = open(filepath1, 'rb').read().decode(encoding='utf-8')
shakespeare_text = open(filepath1, 'rb').read().decode(encoding='utf-8').lower()

filepath2 = "C:/Users/babat/Downloads/Datasets/dataset1.txt"
dataset1_text = open(filepath2, 'rb').read().decode(encoding='utf-8')
dataset1_text = open(filepath2, 'rb').read().decode(encoding='utf-8').lower()

filepath3 = "C:/Users/babat/Downloads/Datasets/dataset2.txt"
dataset2_text = open(filepath3, 'rb').read().decode(encoding='utf-8')
dataset2_text = open(filepath3, 'rb').read().decode(encoding='utf-8').lower()

# Merging all the datasets into one single file used for the training
merger = "\n".join([shakespeare_text, dataset1_text, dataset2_text])

# Shuffle the combined corpus to optimize the training
merger_lines = merger.splitlines()
random.shuffle(merger_lines)
shuffled_merger = "\n".join(merger_lines)

# Limit the number of characters in the array that will be used for the training
shuffled_merger = shuffled_merger[200000:800000]

characters = sorted(set(shuffled_merger))

# Associating each character in the text to an index and vice-versa

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_char = []

for i in range(0, len(shuffled_merger) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(shuffled_merger[i: i + SEQ_LENGTH])
    next_char.append(shuffled_merger[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH,
              len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences),
              len(characters)), dtype=np.bool_)

for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1

# Define the input layer and create the input shape
input_shape = (SEQ_LENGTH, len(characters))
model_input = keras.layers.Input(shape=input_shape)

model = Sequential()
model.add(model_input)
model.add(LSTM(128,
               input_shape=(SEQ_LENGTH,
                            len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

model.fit(x, y, batch_size=512, epochs=24)

# Save the trained model for ulterior use
model.save("Poem_Generator.model")
# Portion to generate poems

# def sample(preds, temperature=1.0):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
#
# def generate_text(length, temperature):
#     start_index = random.randint(0, len(shuffled_merger) - SEQ_LENGTH - 1)
#     generated = ''
#     sentence = shuffled_merger[start_index: start_index + SEQ_LENGTH]
#     generated += sentence
#     for i in range(length):
#         x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
#         for t, char in enumerate(sentence):
#             x_predictions[0, t, char_to_index[char]] = 1
#
#         predictions = model.predict(x_predictions, verbose=0)[0]
#         next_index = sample(predictions,
#                                  temperature)
#         next_character = index_to_char[next_index]
#
#         generated += next_character
#         sentence = sentence[1:] + next_character
#     return generated
