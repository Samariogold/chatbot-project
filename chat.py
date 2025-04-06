import os
import json
import pickle
import random
import traceback
import numpy as np
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize.simple import SpaceTokenizer
from tensorflow.keras.models import load_model

from sheet_service import (
    get_empresas_unicas,
    get_lafiaventuras,
    get_codigo_disponible,
    registrar_aceptacion_usuario
)

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()
tokenizer = SpaceTokenizer()

# Estado del usuario
user_states = {}

# Cargar recursos
try:
    print("📦 Cargando recursos...")
    with open("intents.json") as file:
        intents = json.load(file)
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    model = load_model("chatbot_model.h5")
    print("✅ Recursos cargados correctamente.")
except Exception as e:
    print("❌ Error al cargar recursos:")
    traceback.print_exc()

def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

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
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        threshold = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    except Exception as e:
        print("❌ Error en predicción:")
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

def mostrar_empresas():
    empresas = get_empresas_unicas()
    opciones = "\n".join([f"{i+1}. {e}" for i, e in enumerate(empresas)])
    return f"¡Hola! Soy Lafi 🤖. Estoy aquí para ayudarte a vivir tu próxima Lafiaventura.\n\nPrimero, dime qué empresa o persona deseas explorar. Aquí tienes algunas opciones:\n{opciones}"

def mostrar_lafiaventuras(empresa):
    aventuras = get_lafiaventuras(empresa)
    opciones = "\n".join([f"{i+1}. {a}" for i, a in enumerate(aventuras)])
    return f"Estas son las Lafiaventuras disponibles para *{empresa}*:\n{opciones}\n\nResponde con el número o el nombre de la Lafiaventura que deseas hacer."

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    try:
        msg = request.values.get("Body", "").strip()
        from_number = request.values.get("From")
        user_id = from_number

        print(f"📩 Mensaje recibido de {user_id}: {msg}")

        resp = MessagingResponse()

        if msg.lower() in ["hola", "inicio", "empezar", "reiniciar", "start"]:
            user_states[user_id] = {"stage": "terminos"}
            mensaje = "🤖 Antes de continuar, por favor acepta nuestros Términos y Condiciones para procesar tus datos. Escribe *ACEPTO* para continuar."
            resp.message(mensaje)
            print("➡️ Enviando mensaje:", mensaje)
            return str(resp)

        state = user_states.get(user_id)

        if not state:
            user_states[user_id] = {"stage": "terminos"}
            resp.message("🤖 Antes de continuar, por favor acepta nuestros Términos y Condiciones para procesar tus datos. Escribe *ACEPTO* para continuar.")
            return str(resp)

        elif state["stage"] == "terminos":
            if msg.strip().upper() == "ACEPTO":
                registrar_aceptacion_usuario(user_id)
                user_states[user_id]["stage"] = "empresa"
                resp.message(mostrar_empresas())
                return str(resp)
            else:
                resp.message("Para continuar, por favor escribe *ACEPTO* para aceptar los Términos y Condiciones.")
                return str(resp)

        elif state["stage"] == "empresa":
            empresas = get_empresas_unicas()
            seleccion = msg.lower()

            if seleccion.isdigit() and 1 <= int(seleccion) <= len(empresas):
                empresa = empresas[int(seleccion) - 1]
            else:
                coincidencias = [e for e in empresas if e.lower() == seleccion]
                if not coincidencias:
                    resp.message("No encontré esa empresa/persona. Por favor responde con un número o nombre válido.")
                    return str(resp)
                empresa = coincidencias[0]

            user_states[user_id]["empresa"] = empresa
            user_states[user_id]["stage"] = "lafiaventura"
            resp.message(mostrar_lafiaventuras(empresa))
            return str(resp)

        elif state["stage"] == "lafiaventura":
            empresa = state["empresa"]
            aventuras = get_lafiaventuras(empresa)
            seleccion = msg.lower()

            if seleccion.isdigit() and 1 <= int(seleccion) <= len(aventuras):
                lafiaventura = aventuras[int(seleccion) - 1]
            else:
                coincidencias = [a for a in aventuras if a.lower() == seleccion]
                if not coincidencias:
                    resp.message("No encontré esa Lafiaventura. Por favor responde con un número o nombre válido.")
                    return str(resp)
                lafiaventura = coincidencias[0]

            codigo = get_codigo_disponible(empresa, lafiaventura, user_id)
            if codigo:
                user_states[user_id]["stage"] = "finalizado"
                resp.message(f"🎉 Tu código para la Lafiaventura *{lafiaventura}* es: *{codigo}*.\n\n¡Nos vemos pronto! 🌟")
            else:
                resp.message("😕 Lo siento, no hay más códigos disponibles para esta Lafiaventura. Prueba con otra o intenta más tarde.")
            return str(resp)

        elif state["stage"] == "finalizado":
            resp.message("Si deseas iniciar otra aventura, escribe *Hola* para reiniciar el proceso.")
            return str(resp)

        # NLP fallback
        ints = predict_class(msg)
        res = get_response(ints, intents)
        resp.message(res)
        return str(resp)

    except Exception as e:
        print("❌ Error procesando mensaje:")
        traceback.print_exc()
        resp = MessagingResponse()
        resp.message("Lo siento, ocurrió un error en el bot. Intenta más tarde.")
        return str(resp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)