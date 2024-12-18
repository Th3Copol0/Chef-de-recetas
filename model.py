from datasets import load_dataset

# Cargar el dataset de recetas en español
dataset = load_dataset("somosnlp/recetas-cocina")

# Mostrar las primeras filas de los datos
print(dataset['train'][0])  # Ajusta esto si el dataset tiene una división distinta
import pandas as pd

# Cargar el dataset
data = pd.DataFrame(dataset['train'])

# Seleccionar las primeras 10 recetas del DataFrame
subset_data = data.iloc[0:628]  # Recetas de la 0 a la 9 (10 recetas)

# Imprimir el subconjunto de datos
print(subset_data)


# Mostrar las primeras filas del dataset
print(data.head())

# Mostrar los nombres de las columnas
print(data.columns)
subset_data[subset_data.duplicated(['steps'], keep=False)]
subset_data.drop_duplicates(subset=['steps'], inplace=True)

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization: Converting words into integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(subset_data['steps'])

total_words = len(tokenizer.word_index) + 1

# Creating input sequences
input_sequences = []
for line in subset_data['steps']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding sequences and creating predictors and label
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Defining a generator function for batch processing
def sequence_generator(input_sequences, labels, batch_size):
    num_batches = len(input_sequences) // batch_size
    while True:
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            yield X[start:end, :], y[start:end, :]

# Model definition
model = Sequential()
model.add(Embedding(total_words, 100))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(int(total_words/2), activation='relu'))  # Convert to integer
model.add(Dense(total_words, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model using fit_generator
batch_size = 32
num_epochs = 60
steps_per_epoch = len(X) // batch_size  # Corrected to use len(X) instead of len(input_sequences)

model.fit(
    sequence_generator(X, y, batch_size),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=1
)

# Generating a recipe
def generate_recipe(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example usage
generated_recipe = generate_recipe("Huevo")
print(generated_recipe)
model.save('recipe_generation_model60.keras')

model.save('recipe_generation_model60.h5')

