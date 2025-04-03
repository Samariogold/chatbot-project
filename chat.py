import random
import json
import pickle
import numpy as np
import nltk
import warnings

from flask import Flask, request
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from twilio.twiml.messaging_response import MessagingResponse
from nltk.tokenize.simple import SpaceTokenizer

# Configurar rutas de nltk y desactivar advertencias
nltk.data.path.append('/opt/render/project/src/nltk_data')  # Ruta para entorno en producción
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Inicializar Flask
app = Flask(__name__)

# Cargar recursos del chatbot
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Tokenizador básico para evitar error con punkt
tokenizer = SpaceTokenizer()

# Preprocesamiento de entrada
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predicción de intención
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# Selección de respuesta
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Lo siento, no entiendo tu pregunta."
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Ruta para recibir mensajes de WhatsApp
@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    try:
        msg = request.values.get("Body", "")
        from_number = request.values.get("From", "")

        print(f"📩 Mensaje recibido de {from_number}: {msg}")

        intents_list = predict_class(msg)
        response_text = get_response(intents_list, intents)

        print(f"🤖 Enviando respuesta: {response_text}")

        resp = MessagingResponse()
        resp.message(response_text)
        return str(resp)

    except Exception as e:
        print(f"🚨 Error en /whatsapp: {e}")
        resp = MessagingResponse()
        resp.message("Lo siento, ocurrió un error en el bot. Intenta más tarde.")
        return str(resp)

# Ruta raíz opcional
@app.route("/")
def index():
    return "🤖 El chatbot está activo."

