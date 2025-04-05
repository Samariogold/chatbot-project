import os
import json
import gspread
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# Autorizar acceso a Google Sheets
def autorizar_gspread():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_info = json.loads(os.getenv("GOOGLE_CREDENTIALS_JSON"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
    client = gspread.authorize(creds)
    return client

# Leer todos los datos de la hoja
def get_lafi_data():
    client = autorizar_gspread()
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit").sheet1
    return sheet.get_all_records()

# Empresas/personas únicas
def get_empresas_unicas():
    data = get_lafi_data()
    return sorted(set(row['Empresa/persona'] for row in data if row['Empresa/persona']))

# Lafiaventuras disponibles por empresa
def get_lafiaventuras(empresa):
    data = get_lafi_data()
    return sorted(set(
        row['Lafiaventura'] for row in data
        if row['Empresa/persona'].lower() == empresa.lower()
    ))

# Buscar y registrar código disponible
def get_codigo_disponible(empresa, lafiaventura, numero_usuario):
    client = autorizar_gspread()
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit").sheet1
    data = sheet.get_all_records()

    for i, row in enumerate(data):
        if (
            row['Empresa/persona'].lower() == empresa.lower()
            and row['Lafiaventura'].lower() == lafiaventura.lower()
            and not row.get('Usado', '').strip()
        ):
            fila = i + 2  # +2 por encabezado
            sheet.update_cell(fila, 4, "Sí")  # Columna "Usado"
            sheet.update_cell(fila, 5, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # "Fecha de asignación"
            sheet.update_cell(fila, 6, numero_usuario)  # "Usuario WhatsApp"
            return row['Codigo']
    
    return None

# Registro de aceptación de Términos y Condiciones
def registrar_aceptacion_usuario(numero_usuario):
    client = autorizar_gspread()
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit").sheet1
    sheet.append_row(["", "", "", "Sí", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), numero_usuario])