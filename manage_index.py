from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Inicializar la conexión
# Reemplaza con tus credenciales de app.pinecone.io
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY_MANAGE")
)

# 2. Listar índices actuales
print("Índices actuales:", pc.list_indexes().names())

# 3. Crear un índice básico
# 'dimension' debe coincidir con el tamaño de tus vectores/embeddings
if "example-index" not in pc.list_indexes().names():
    pc.create_index(
        name="example-index", 
        dimension=128,
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print("Índice 'example-index' creado.")

# 4. Crear un índice con configuración avanzada
if "second-index" not in pc.list_indexes().names():
    pc.create_index(
        name="second-index", 
        dimension=128, 
        metric="euclidean",  # Opciones: 'cosine', 'dotproduct', 'euclidean'
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
        # Notas: shards y replicas se manejan automáticamente en Serverless
    )
    print("Índice 'second-index' creado.")

# 5. Ver detalles de un índice
index_description = pc.describe_index("example-index")
print("\nDescripción del índice:")
print(index_description)

# 6. Eliminar un índice
# pc.delete_index("example-index")
# print("\nÍndice 'example-index' eliminado.")

# 7. Listar de nuevo para confirmar
print("\nLista final de índices:", pc.list_indexes().names())