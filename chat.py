import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Configurar rutas para evitar problemas con 'punkt_tab'
nltk.data.path.append('/Users/ronaldalfaro/nltk_data')

# Ignorar advertencias de SSL
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Cargar el modelo entrenado y los datos
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# ✅ Importar SpaceTokenizer para evitar `punkt_tab`
from nltk.tokenize.simple import SpaceTokenizer

tokenizer = SpaceTokenizer()

# ✅ Función para limpiar y procesar la oración
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)  # ✅ SpaceTokenizer en lugar de word_tokenize
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# ✅ Crear una bolsa de palabras (bag of words)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# ✅ Predecir la clase/intención
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    
    # Ordenar resultados por probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# ✅ Obtener la respuesta basada en la intención detectada
def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Lo siento, no entiendo tu pregunta."
    
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# ✅ Iniciar el chatbot
print("🤖 Chatbot activado. Escribe 'salir' para terminar.")

while True:
    message = input("Tú: ")
    if message.lower() == 'salir':
        print("🤖 Chatbot: ¡Adiós!")
        break
    
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(f"🤖 Chatbot: {res}")

