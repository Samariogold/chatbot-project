import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Configuración de acceso a Google Sheets
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

# Asegúrate de que este archivo exista en tu carpeta del proyecto
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)

# URL de tu hoja de cálculo de códigos
SHEET_URL = "https://docs.google.com/spreadsheets/d/1gWlNtrhecTkaPzVT4Fj_ujcC74aNNn_qosJ5UhPejLM/edit"

def get_lafi_data():
    """Obtiene todos los registros como una lista de diccionarios"""
    sheet = client.open_by_url(SHEET_URL).sheet1
    data = sheet.get_all_records()
    return data

def get_codigo_disponible(empresa, lafiaventura):
    """Entrega un código libre (no usado) para la empresa y Lafiaventura, y lo marca como usado"""
    sheet = client.open_by_url(SHEET_URL).sheet1
    records = sheet.get_all_records()

    for i, row in enumerate(records):
        if (
            row['Empresa/persona'].strip().lower() == empresa.strip().lower() and
            row['Lafiaventura'].strip().lower() == lafiaventura.strip().lower() and
            not row.get('Usado', '').strip()
        ):
            # Marca como usado en la hoja (sumamos 2 porque get_all_records() omite encabezado)
            sheet.update_cell(i + 2, 4, "Sí")
            return row['Codigo']

    return None
