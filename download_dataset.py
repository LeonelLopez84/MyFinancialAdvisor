import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración extraída de la imagen
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = 'anyoneai-datasets'
PREFIX = 'nasdaq_annual_reports/' # La carpeta dentro del bucket

# Crear sesión con las credenciales
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name='us-east-1'  # <--- AGREGA ESTO
)

s3 = session.resource('s3')
bucket = s3.Bucket(BUCKET_NAME)

# Crear carpeta local para guardar los archivos
if not os.path.exists('descargas'):
    os.makedirs('descargas')

print(f"Listando archivos en {BUCKET_NAME}/{PREFIX}...")

# Iterar sobre los objetos y descargar
# NOTA: Son muchas empresas (2600+), este script descargará todo.
for obj in bucket.objects.filter(Prefix=PREFIX):
    # Obtener el nombre del archivo (quitando el prefijo de carpeta si es necesario)
    filename = obj.key.split('/')[-1]
    
    if filename: # Evitar descargar la propia carpeta como archivo
        local_path = os.path.join('descargas', filename)
        if os.path.exists(local_path):
            print(f"Saltando {filename} (ya existe)")
        else:
            print(f"Descargando: {filename}")
            bucket.download_file(obj.key, local_path)

print("Descarga completada.")