import os
import random
import numpy as np
import tensorflow as tf
import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.layers import Activation, Dense, LSTM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import the training data from the different datasets used for our model
# by opening the file then converting the text in each file  into numbers

filepath1 = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# shakespeare_text = open(filepath1, 'rb').read().decode(encoding='utf-8')
shakespeare_text = open(filepath1, 'rb').read().decode(encoding='utf-8').lower()

filepath2 = "C:/Users/babat/Downloads/Datasets/dataset1.txt"
# dataset1_text = open(filepath2, 'rb').read().decode(encoding='utf-8')
dataset1_text = open(filepath2, 'rb').read().decode(encoding='utf-8').lower()

filepath3 = "C:/Users/babat/Downloads/Datasets/dataset2.txt"
# dataset2_text = open(filepath3, 'rb').read().decode(encoding='utf-8')
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


#Load the trained model
model = tf.keras.models.load_model('Poem_Generator.keras')


# Portion to generate poems

# Sample code copied from the official Keras tutorial platform
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, temperature):
    max_len = len(shuffled_merger) - SEQ_LENGTH
    start_index = random.randint(0, max_len)
    # Assuming hidden and cell state have a dimensionality of 128 (as seen in arguments)
    initial_state = (tf.zeros_like(model.layers[0].state_size[0]),
                     tf.zeros_like(model.layers[0].state_size[1]))

    generated = ''
    sentence = shuffled_merger[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        lstm_layer = model.layers[0]
        lstm_layer.reset_states()
        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
    
        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

# Print the layer names in your model
# layer_names = [layer.name for layer in model.layers]
# print("Layer Names:", layer_names)


print("---------------------------Temp == 0.2---------------------------")
print(generate_text(300, 0.2))

'''
print("---------------------------Temp==0.4---------------------------")
print(generate_text(300, 0.4))
print("---------------------------Temp==0.5---------------------------")
print(generate_text(300, 0.5))
print("---------------------------Temp==0.6---------------------------")
print(generate_text(300, 0.6))
print("---------------------------Temp==0.8---------------------------")
print(generate_text(300, 0.8))
print("---------------------------Temp==1.0---------------------------")
print(generate_text(300, 1.0))
'''
