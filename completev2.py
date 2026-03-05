import os
import torch
import fitz # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import sqlite_vec
import json
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. CONFIGURACIÓN DE LOCAL
LOCAL_DIR = 'descargas'
DB_PATH = 'vectordb.sqlite'

# 2. INICIALIZACIÓN DE MODELOS Y CLIENTES
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Usando dispositivo: {device.upper()} ---")

model = SentenceTransformer('BAAI/bge-m3', device=device)
if device == 'cuda':
    model.half() # Convertir a float16 para RTX 3060

# Configurar SQLite Vectorial
print(f"--- Creando/Conectando Base de Datos Vectorial ({DB_PATH}) ---")
db = sqlite3.connect(DB_PATH)
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Crear tablas
db.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT UNIQUE,
    text TEXT,
    source TEXT,
    folder TEXT
)
''')

# Crear tabla virtual para vectores sqlite-vec
try:
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vector_index USING vec0(embedding float[1536])")
except sqlite3.OperationalError:
    # Si la tabla ya existe y falla el IF NOT EXISTS en algunas versiones de SQLite, lo ignoramos
    pass
db.commit()


# Configuración del Splitter para los reportes (Documentos largos)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

def procesar_y_cargar():
    if not os.path.exists(LOCAL_DIR):
        print(f"❌ La carpeta '{LOCAL_DIR}' no se encontró.")
        return
        
    archivos = [f for f in os.listdir(LOCAL_DIR) if os.path.isfile(os.path.join(LOCAL_DIR, f))]

    # Consultar qué archivos ya están en la base de datos para no repetirlos
    archivos_procesados = set()
    try:
        cur = db.cursor()
        for row in cur.execute("SELECT DISTINCT source FROM documents"):
            archivos_procesados.add(row[0])
    except sqlite3.OperationalError:
        pass
        
    archivos_pendientes = [f for f in archivos if f not in archivos_procesados]
    print(f"--- Total de archivos: {len(archivos)} | Ya procesados: {len(archivos_procesados)} | Pendientes: {len(archivos_pendientes)} ---")

    for filename in tqdm(archivos_pendientes, desc=f"Procesando desde Disco Duro ({LOCAL_DIR})"):
        filepath = os.path.join(LOCAL_DIR, filename)

        # 1. Leer archivo desde disco local
        try:
            if filename.lower().endswith('.pdf'):
                body = ""
                with fitz.open(filepath) as doc:
                    for page in doc:
                        body += page.get_text()
            else:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file_obj:
                    body = file_obj.read()
        except Exception as e:
            print(f"Error al procesar {filename}: {e}")
            continue

        # 2. Chunking (Fragmentación)
        chunks = text_splitter.split_text(body)
        
        # 3. Embedding y Subida por bloques
        batch_size = 256
        
        with db:
            cur = db.cursor()
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Generar vectores usando la GPU
                embeddings = model.encode(batch_chunks, show_progress_bar=False, batch_size=64)
                
                # RELLENO A 1536 (debe coincidir con la dimensión definida)
                padded_embeddings = np.pad(embeddings, ((0, 0), (0, 1536 - embeddings.shape[1])), mode='constant').astype(np.float32)
                
                for j, emb in enumerate(padded_embeddings):
                    vector_id = f"{filename.replace(' ', '_')}_{i+j}"
                    text = batch_chunks[j]
                    
                    # Insertar o reemplazar metadata
                    cur.execute(
                        "INSERT OR REPLACE INTO documents (chunk_id, text, source, folder) VALUES (?, ?, ?, ?)", 
                        (vector_id, text, filename, LOCAL_DIR)
                    )
                    
                    # Recuperar el rowid correcto
                    cur.execute("SELECT id FROM documents WHERE chunk_id = ?", (vector_id,))
                    row_id = cur.fetchone()[0]
                    
                    # Insertar o reemplazar vector usando numpy array (tobytes() produce el formato adecuado de float32 raw)
                    emb_bytes = emb.tobytes() 
                    
                    # Para virtual tables de sqlite-vec, si el registro ya existe, un INSERT directo en vec0 puede dar UNIQUE constraint failed
                    # Lo recomendado es borrar primero si existe, y luego insertar
                    cur.execute("DELETE FROM vector_index WHERE rowid = ?", (row_id,))
                    cur.execute(
                        "INSERT INTO vector_index (rowid, embedding) VALUES (?, ?)", 
                        (row_id, emb_bytes)
                    )

if __name__ == "__main__":
    procesar_y_cargar()
    
    # Prueba rápida
    count = db.execute("SELECT count(*) FROM vector_index").fetchone()[0]
    print(f"✅ Carga finalizada exitosamente. Total vectores en DB: {count}")
    
    db.close()
