import os
import torch
import fitz # PyMuPDF
import numpy as np
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import itertools

from dotenv import load_dotenv
load_dotenv()

# 1. CONFIGURACIÓN DE LOCAL
LOCAL_DIR = 'descargas'

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY_COMPLETE")
INDEX_NAME = "nasdaq-index-1536" # Asegúrate que sea de 1536 dimensiones

# 2. INICIALIZACIÓN DE MODELOS Y CLIENTES
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Usando dispositivo: {device.upper()} ---")

model = SentenceTransformer('BAAI/bge-m3', device=device)
if device == 'cuda':
    model.half() # Convertir a float16 para RTX 3060


pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"--- Creando índice: {INDEX_NAME} ---")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536, 
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"--- Índice {INDEX_NAME} creado exitosamente ---")

index = pc.Index(INDEX_NAME)

# Cliente de S3 removido ya que leemos desde disco


# Configuración del Splitter para los reportes (Documentos largos)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

def procesar_y_cargar():
    # Listar objetos en disco duro local
    if not os.path.exists(LOCAL_DIR):
        print(f"❌ La carpeta '{LOCAL_DIR}' no se encontró.")
        return
        
    archivos = [f for f in os.listdir(LOCAL_DIR) if os.path.isfile(os.path.join(LOCAL_DIR, f))]

    for filename in tqdm(archivos, desc=f"Procesando desde Disco Duro ({LOCAL_DIR})"):
        filepath = os.path.join(LOCAL_DIR, filename)

        # 1. Leer archivo desde disco local (rapidísimo)
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
        
        # 3. Embedding y Subida por bloques (GPU OPTIMIZADO)
        # Para una RTX 3060 de 12GB, 256 o 512 chunks suele ser ideal para exprimir los CUDA cores
        batch_size = 256
        
        all_vectors_for_file = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Generar vectores usando la GPU
            embeddings = model.encode(batch_chunks, show_progress_bar=False, batch_size=64)
            
            # RELLENO A 1536 PARA CUMPLIR REQUISITO PINECONE
            padded_embeddings = np.pad(embeddings, ((0, 0), (0, 1536 - embeddings.shape[1])), mode='constant')
            
            for j, emb in enumerate(padded_embeddings):
                vector_id = f"{filename.replace(' ', '_')}_{i+j}"
                all_vectors_for_file.append({
                    "id": vector_id,
                    "values": emb.tolist(),
                    "metadata": {
                        "text": batch_chunks[j],
                        "source": filename,
                        "folder": LOCAL_DIR
                    }
                })
        
        # Subir a Pinecone en bloques asíncronos para no trabar la generación
        # Pinecone recomienda batches de 100-200 vectores por request
        def chunks_of_list(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        # Enviamos a Pinecone (conteo de subida)
        for chunk_of_vectors in chunks_of_list(all_vectors_for_file, 200):
             # upsert(async_req=False) es default, pero limitando a 200 va como bala
             index.upsert(vectors=chunk_of_vectors)

if __name__ == "__main__":
    procesar_y_cargar()
    print("✅ Carga finalizada exitosamente.")