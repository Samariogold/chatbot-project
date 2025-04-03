import gspread
import os
import json
from oauth2client.service_account import ServiceAccountCredentials

def get_lafi_data():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_info = json.loads(os.getenv("GOOGLE_CREDENTIALS_JSON"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit").sheet1
    data = sheet.get_all_records()
    return data

def get_codigo_disponible(empresa, lafiaventura):
    data = get_lafi_data()
    for i, row in enumerate(data):
        if (row['Empresa/persona'].lower() == empresa.lower()
                and row['Lafiaventura'].lower() == lafiaventura.lower()
                and not row.get('Usado', '').strip()):
            marcar_codigo_como_usado(row['Codigo'])
            return row['Codigo']
    return None

def marcar_codigo_como_usado(codigo):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_info = json.loads(os.getenv("GOOGLE_CREDENTIALS_JSON"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit").sheet1
    data = sheet.get_all_records()
    for i, row in enumerate(data):
        if row['Codigo'] == codigo:
            sheet.update_cell(i + 2, 4, "Sí")  # Fila +1 por encabezado, columna 4 = "Usado"
            break
