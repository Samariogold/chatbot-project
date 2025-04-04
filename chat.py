import random
import json
import pickle
import numpy as np
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import nltk
import traceback

# Descargas necesarias de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# Cargar los archivos necesarios
try:
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
    print("✅ Modelo y archivos cargados correctamente")
except Exception as e:
    print("❌ Error cargando modelo o archivos:")
    traceback.print_exc()

# Funciones del bot
def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
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

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Lo siento, no entiendo tu pregunta."
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Lo siento, no tengo respuesta para eso."

# Ruta principal para WhatsApp
@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    try:
        msg = request.values.get("Body", "")
        from_number = request.values.get("From")

        print("📩 Mensaje recibido de WhatsApp:", from_number, msg)
        print("✅ Iniciando predicción...")

        ints = predict_class(msg)
        print("🔍 Intenciones detectadas:", ints)

        res = get_response(ints, intents)
        print("🤖 Respuesta generada:", res)

    except Exception as e:
        print("❌ Error capturado en /whatsapp:")
        traceback.print_exc()
        res = "Lo siento, ocurrió un error en el bot. Intenta más tarde."

    resp = MessagingResponse()
    resp.message(res)
    return str(resp)

# Para pruebas locales
if __name__ == "__main__":
    app.run(debug=True)