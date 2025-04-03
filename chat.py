import os
import json
import pickle
import random
import numpy as np
from flask import Flask, request
from twilio.rest import Client
from dotenv import load_dotenv
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.simple import SpaceTokenizer
from tensorflow.keras.models import load_model

# Inicializar Flask
app = Flask(__name__)

# Cargar variables de entorno
load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# NLP y modelo
lemmatizer = WordNetLemmatizer()
tokenizer = SpaceTokenizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

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
    if not intents_list:
        return "Lo siento, no entendí tu mensaje."
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    msg = request.values.get("Body", "")
    from_number = request.values.get("From")
    
    ints = predict_class(msg)
    res = get_response(ints, intents)
    
    client.messages.create(
        body=res,
        from_=TWILIO_PHONE_NUMBER,
        to=from_number
    )
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "✅ LafiBot en Flask está activo."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
