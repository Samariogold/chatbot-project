import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request
from tensorflow.keras.models import load_model
from twilio.twiml.messaging_response import MessagingResponse

# Descargar recursos necesarios de NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializar Flask
app = Flask(__name__)

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

# Cargar archivos del modelo entrenado
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

# Funciones auxiliares
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Lo siento, no entendí tu mensaje."
    
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Lo siento, no tengo una respuesta para eso."

# Ruta de recepción desde Twilio
@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    try:
        msg = request.values.get("Body", "")
        from_number = request.values.get("From")
        print(f"📩 Mensaje recibido de {from_number}: {msg}")

        ints = predict_class(msg)
        res = get_response(ints, intents)

    except Exception as e:
        print(f"❌ Error en el bot: {e}")
        res = "Lo siento, ocurrió un error en el bot. Intenta más tarde."

    resp = MessagingResponse()
    resp.message(res)
    return str(resp)

# Ruta base (opcional)
@app.route("/", methods=["GET"])
def home():
    return "🤖 LafiBot está activo."

