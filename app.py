import os
import json
import pickle
import random
import numpy as np
import time
from flask import Flask, request
from twilio.rest import Client
from dotenv import load_dotenv
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.simple import SpaceTokenizer
from tensorflow.keras.models import load_model
from sheet_service import get_lafi_data, get_codigo_disponible

# Cargar variables de entorno
load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
app = Flask(__name__)

# Cargar modelo NLP
lemmatizer = WordNetLemmatizer()
tokenizer = SpaceTokenizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Estados por usuario
user_states = {}
last_interaction = {}

def responder(texto, to):
    client.messages.create(body=texto, from_=TWILIO_PHONE_NUMBER, to=to)
    return "OK", 200

def get_enterprises():
    data = get_lafi_data()
    return sorted(list(set(row['Empresa/persona'] for row in data)))

def get_unique_lafiaventuras(empresa):
    data = get_lafi_data()
    aventuras = [row['Lafiaventura'] for row in data if row['Empresa/persona'].lower() == empresa.lower()]
    return sorted(list(set(aventuras)))

@app.route('/whatsapp', methods=['POST'])
def whatsapp():
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From')

    if not from_number:
        return "Error: No se recibió el número 'From' desde Twilio", 400

    ahora = time.time()
    if from_number in last_interaction and ahora - last_interaction[from_number] > 300:
        user_states[from_number] = 'inicio'
    last_interaction[from_number] = ahora

    state = user_states.get(from_number, 'inicio')

    if incoming_msg.lower() in ['inicio', 'empezar', 'reiniciar', 'salir']:
        state = 'inicio'

    if state == 'inicio':
        user_states[from_number] = 'esperando_empresa'
        empresas = get_enterprises()
        empresas_str = "\n".join(f"{i+1}. {e}" for i, e in enumerate(empresas))
        user_states[from_number + '_empresas'] = empresas
        return responder(
            f"¡Hola! Soy Lafi 🤖✨\nEstas son las empresas/personas con Lafiaventuras:\n\n{empresas_str}\n\nPor favor, escribe el número o el nombre de una empresa/persona para continuar.",
            from_number
        )

    elif state == 'esperando_empresa':
        empresas = user_states.get(from_number + '_empresas', get_enterprises())
        seleccion = incoming_msg
        if seleccion.isdigit() and 1 <= int(seleccion) <= len(empresas):
            empresa = empresas[int(seleccion)-1]
        elif seleccion in empresas:
            empresa = seleccion
        else:
            return responder("No reconozco esa empresa. Por favor, elige una opción válida.", from_number)

        user_states[from_number] = {'estado': 'esperando_lafiaventura', 'empresa': empresa}
        aventuras = get_unique_lafiaventuras(empresa)
        user_states[from_number + '_aventuras'] = aventuras
        aventuras_str = "\n".join(f"{i+1}. {a}" for i, a in enumerate(aventuras))
        return responder(
            f"Perfecto. Estas son las Lafiaventuras que ofrece *{empresa}*:\n\n{aventuras_str}\n\nEscribe el número o nombre de la que te interesa.",
            from_number
        )

    elif isinstance(state, dict) and state.get('estado') == 'esperando_lafiaventura':
        empresa = state['empresa']
        aventuras = user_states.get(from_number + '_aventuras', [])
        seleccion = incoming_msg
        if seleccion.isdigit() and 1 <= int(seleccion) <= len(aventuras):
            lafiaventura = aventuras[int(seleccion)-1]
        elif seleccion in aventuras:
            lafiaventura = seleccion
        else:
            return responder("No encontré esa Lafiaventura. Intenta con otra de la lista.", from_number)

        codigo = get_codigo_disponible(empresa, lafiaventura)
        if codigo:
            user_states[from_number] = 'inicio'
            return responder(
                f"¡Genial! 😄\nTu código para la Lafiaventura *{lafiaventura}* de *{empresa}* es:\n\n👉 *{codigo}*\n\n¿Quieres consultar otra empresa? Escribe 'inicio'.",
                from_number
            )
        else:
            return responder("😕 Lo siento, ya no hay códigos disponibles para esa Lafiaventura.", from_number)

    else:
        return responder("No entendí tu mensaje. Escribe 'inicio' para empezar de nuevo.", from_number)
