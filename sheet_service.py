import gspread
import os
import json
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# Acceso a Google Sheets
def autorizar_gspread():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_info = json.loads(os.getenv("GOOGLE_CREDENTIALS_JSON"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
    client = gspread.authorize(creds)
    return client

# Obtener todas las filas
def get_lafi_data():
    client = autorizar_gspread()
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit").sheet1
    data = sheet.get_all_records()
    return data

# Obtener empresas únicas
def get_empresas():
    data = get_lafi_data()
    return sorted(set(row['Empresa/persona'] for row in data if row['Empresa/persona']))

# Obtener aventuras únicas de una empresa
def get_lafiaventuras(empresa):
    data = get_lafi_data()
    return sorted(set(row['Lafiaventura'] for row in data if row['Empresa/persona'].lower() == empresa.lower()))

# Obtener código disponible y marcar como usado
def get_codigo_disponible(empresa, lafiaventura, numero_usuario):
    client = autorizar_gspread()
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit").sheet1
    data = sheet.get_all_records()

    for i, row in enumerate(data):
        if (row['Empresa/persona'].lower() == empresa.lower()
            and row['Lafiaventura'].lower() == lafiaventura.lower()
            and not row.get('Usado', '').strip()):

            # Marcar como usado
            fila = i + 2  # sumamos 2 por el encabezado
            sheet.update_cell(fila, 4, "Sí")  # Columna "Usado"
            sheet.update_cell(fila, 5, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # Columna "Fecha de asignación"
            sheet.update_cell(fila, 6, numero_usuario)  # Columna "Usuario WhatsApp"
            return row['Codigo']

    return None

# Registrar aceptación de términos
def registrar_aceptacion_usuario(numero_usuario):
    client = autorizar_gspread()
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit").sheet1
    sheet.append_row(["", "", "", "Sí", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), numero_usuario])