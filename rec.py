from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

app = Flask(__name__)
from datasets import load_dataset
recetas_df = pd.read_csv('recetas2.csv')
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
# Loading the saved model
model = tf.keras.models.load_model('recipe_generation_model60.h5')

# Loading the dataset
#data = pd.read_csv('recetas.csv')

#data.drop_duplicates(subset=['steps'], inplace=True)

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
# Maximum sequence length
max_sequence_length = max([len(seq) for seq in input_sequences])

# Function to generate the recipe
def generate_recipe(seed_text, next_words=99):
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

# Defining the home page of our site
@app.route('/')
def index():
    return render_template('index.html', recipes=recetas_df.to_dict(orient='records'))



@app.route('/generate_recipe', methods=['POST'])
def generate():
    user_input = request.form['user_input']
    generated_recipe = generate_recipe(user_input)
    return render_template('index.html', user_input=user_input, generated_recipe=generated_recipe, recipes=recetas_df.to_dict(orient='records'))

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)