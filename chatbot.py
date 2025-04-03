import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Configurar rutas para evitar problemas con 'punkt_tab'
nltk.data.path.append('/Users/ronaldalfaro/nltk_data')

# Ignorar advertencias de SSL
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Descargar las librerías necesarias si aún no están disponibles
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Cargar los datos desde intents.json
with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# ✅ Importar manualmente el tokenizador sin buscar en `punkt_tab`
from nltk.tokenize.simple import SpaceTokenizer

tokenizer = SpaceTokenizer()

# Procesar los patrones y respuestas
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = tokenizer.tokenize(pattern)  # ✅ Uso de SpaceTokenizer
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizar y ordenar las palabras
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Guardar palabras y clases para usarlas en el modelo
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Crear los datos de entrenamiento
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezclar datos para mejorar el rendimiento
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Configurar el optimizador
sgd = SGD(learning_rate=0.01, momentum=0.9, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Guardar el modelo entrenado
model.save('chatbot_model.h5')

print("✅ Modelo entrenado y guardado correctamente.")


