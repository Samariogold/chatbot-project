from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import random
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from nltk.tokenize.simple import SpaceTokenizer

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
tokenizer = SpaceTokenizer()

# Cargar recursos
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
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Lo siento, no entendí tu mensaje."
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    try:
        msg = request.values.get("Body", "")
        from_number = request.values.get("From")

        print(f"📩 Mensaje recibido de {from_number}: {msg}")

        ints = predict_class(msg)
        res = get_response(ints, intents)

        response = MessagingResponse()
        response.message(res)
        return str(response)

    except Exception as e:
        print(f"❌ Error procesando mensaje: {e}")
        response = MessagingResponse()
        response.message("Lo siento, ocurrió un error en el bot. Intenta más tarde.")
        return str(response)

if __name__ == "__main__":
    app.run()
