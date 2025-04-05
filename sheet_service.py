import gspread
import os
import json
from oauth2client.service_account import ServiceAccountCredentials

# Leer las credenciales desde una variable de entorno que contiene el JSON como string
def get_credentials():
    try:
        creds_str = os.getenv("GOOGLE_CREDENTIALS_JSON")
        creds_info = json.loads(creds_str)
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        return ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
    except Exception as e:
        print("❌ Error cargando las credenciales:")
        print(e)
        raise

# Obtener los datos de la hoja
def get_lafi_data():
    creds = get_credentials()
    client = gspread.authorize(creds)
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit").sheet1
    data = sheet.get_all_records()
    return data

# Buscar un código disponible y marcarlo como usado
def get_codigo_disponible(empresa, lafiaventura):
    data = get_lafi_data()
    for i, row in enumerate(data):
        if (
            row['Empresa/persona'].strip().lower() == empresa.strip().lower()
            and row['Lafiaventura'].strip().lower() == lafiaventura.strip().lower()
            and not row.get('Usado', '').strip()
        ):
            marcar_codigo_como_usado(row['Codigo'])
            return row['Codigo']
    return None

# Marcar código como "usado" en la hoja de cálculo
def marcar_codigo_como_usado(codigo):
    creds = get_credentials()
    client = gspread.authorize(creds)
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit").sheet1
    data = sheet.get_all_records()
    for i, row in enumerate(data):
        if row['Codigo'] == codigo:
            sheet.update_cell(i + 2, 4, "Sí")  # Fila +1 por encabezado, columna 4 = "Usado"
            break
