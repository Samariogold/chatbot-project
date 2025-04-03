import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from tensorflow.keras.models import load_model

# Inicializar Flask
app = Flask(__name__)

# Configurar NLTK
nltk.data.path.append('/opt/render/project/src/nltk_data')  # Ruta válida en Render

# Inicializar utilidades de NLP
lemmatizer = WordNetLemmatizer()

# Cargar archivos entrenados
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Tokenizador básico sin punkt
from nltk.tokenize.simple import SpaceTokenizer
tokenizer = SpaceTokenizer()

# Limpieza de oración
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Crear bolsa de palabras
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecir intención
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Obtener respuesta
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Lo siento, no entendí tu mensaje. ¿Puedes reformularlo?"
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Endpoint para WhatsApp
@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    msg = request.values.get("Body", "")
    from_number = request.values.get("From", "")

    print(f"📩 Mensaje recibido de {from_number}: {msg}")

    intents_list = predict_class(msg)
    response_text = get_response(intents_list, intents)

    resp = MessagingResponse()
    resp.message(response_text)

    return str(resp)

# Ejecutar localmente (opcional para pruebas)
if __name__ == "__main__":
    app.run(debug=True)
