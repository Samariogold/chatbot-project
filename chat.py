import os
import json
import pickle
import random
import traceback
import numpy as np
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# NLTK
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

try:
    print("📦 Cargando archivos...")
    with open("intents.json") as file:
        intents = json.load(file)
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    model = load_model("chatbot_model.h5")
    print("✅ Recursos cargados correctamente.")
except Exception as e:
    print("❌ Error cargando archivos:")
    traceback.print_exc()

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
    try:
        print("🧠 Iniciando predicción...")
        bow = bag_of_words(sentence)
        print(f"📊 Array para el modelo: {bow}")
        res = model.predict(np.array([bow]))[0]
        print(f"📈 Resultados del modelo: {res}")
        threshold = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    except Exception as e:
        print("❌ Error en predict_class:")
        traceback.print_exc()
        return []

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Lo siento, no entendí tu mensaje."
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Lo siento, no tengo respuesta para eso."

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    try:
        msg = request.values.get("Body", "")
        from_number = request.values.get("From")
        print(f"📩 Mensaje recibido de WhatsApp: {from_number}: {msg}")

        ints = predict_class(msg)
        res = get_response(ints, intents)
    except Exception as e:
        print("❌ Error capturado en /whatsapp:")
        traceback.print_exc()
        res = "Lo siento, ocurrió un error en el bot. Intenta más tarde."

    resp = MessagingResponse()
    resp.message(res)
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)