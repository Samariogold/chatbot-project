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
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize.simple import SpaceTokenizer
from tensorflow.keras.models import load_model

# Importar servicio de Google Sheets
from sheet_service import get_lafi_data

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()
tokenizer = SpaceTokenizer()

# Estados por usuario
user_states = {}
last_interaction = {}

# Cargar archivos
try:
    print("📦 Cargando archivos...")
    with open("intents.json") as file:
        intents = json.load(file)
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    model = load_model("chatbot_model.h5")
    print("✅ Recursos cargados correctamente.")
except Exception as e:
    print("❌ Error cargando archivos del bot:")
    traceback.print_exc()

# Preprocesamiento
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    print(f"🧹 Palabras procesadas: {sentence_words}")
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    print(f"📊 Array para el modelo: {bag}")
    return np.array(bag)

def predict_class(sentence):
    try:
        print("🧠 Iniciando predicción...")
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        print(f"📈 Resultados del modelo: {res}")
        threshold = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        predictions = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
        print(f"🤖 Predicción: {predictions}")
        return predictions
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

def get_enterprises():
    data = get_lafi_data()
    empresas = sorted(set(row['Empresa/persona'] for row in data))
    return empresas

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    try:
        msg = request.values.get("Body", "").strip()
        from_number = request.values.get("From")

        print(f"📩 Mensaje recibido de {from_number}: {msg}")

        ints = predict_class(msg)
        intent = ints[0]['intent'] if ints else "desconocido"

        # Si es saludo, iniciamos el flujo guiado con lista dinámica
        if intent == "saludo":
            user_states[from_number] = "esperando_empresa"
            empresas = get_enterprises()
            empresas_str = "\n".join(f"{i+1}. {e}" for i, e in enumerate(empresas))
            user_states[from_number + '_empresas'] = empresas
            respuesta = get_response(ints, intents).replace("[Espera la lista...]", empresas_str)
            resp = MessagingResponse()
            resp.message(respuesta)
            return str(resp)

        else:
            # Respuestas estándar si no es flujo guiado
            res = get_response(ints, intents)
            resp = MessagingResponse()
            resp.message(res)
            return str(resp)

    except Exception as e:
        print("❌ Error procesando el mensaje:")
        traceback.print_exc()
        resp = MessagingResponse()
        resp.message("Lo siento, ocurrió un error en el bot. Intenta más tarde.")
        return str(resp)

# Producción en Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)