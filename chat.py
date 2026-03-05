import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

import os
from dotenv import load_dotenv
load_dotenv()

# --- 1. CONFIGURACIÓN (Usa tus mismas credenciales) ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY_CHAT")
INDEX_NAME = "nasdaq-index"

# --- 2. CARGA DE MODELOS ---
print("--- Cargando modelos en GPU... (esto puede tardar unos segundos) ---")

# A. Modelo de Embeddings (El mismo que usaste para cargar)
# BGE-M3 ocupa ~2GB de VRAM
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer('BAAI/bge-m3', device=device)

# B. Conexión a Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# C. Modelo LLM (Llama 3 8B Instruct) en 4 bits
# Esto ocupará unos ~6GB de VRAM, permitiendo que conviva con BGE-M3
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Creamos un pipeline de generación de texto
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # Longitud máxima de la respuesta
    temperature=0.1,    # Baja temperatura para ser preciso en datos financieros
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print("✅ Modelos cargados. Listo para chatear.")

# --- 3. FUNCIONES RAG ---

def recuperar_contexto(pregunta, top_k=5):
    """Convierte la pregunta a vector y busca en Pinecone"""
    # Generar vector de la pregunta
    query_vector = embed_model.encode(pregunta).tolist()
    
    # Buscar en Pinecone
    resultados = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extraer texto limpio
    contextos = []
    for match in resultados['matches']:
        texto = match['metadata']['text']
        fuente = match['metadata'].get('source', 'Desconocido')
        contextos.append(f"[Fuente: {fuente}]\n{texto}")
    
    return contextos

def generar_respuesta(pregunta):
    # 1. Recuperar información relevante
    contextos = recuperar_contexto(pregunta)
    bloque_contexto = "\n\n---\n\n".join(contextos)
    
    # 2. Crear el prompt con formato Llama 3
    prompt_sistema = """Eres un analista financiero experto y preciso. 
    Usa EXCLUSIVAMENTE el siguiente contexto para responder la pregunta del usuario. 
    Si la respuesta no está en el contexto, di "No tengo información suficiente en los documentos".
    Cita la fuente (nombre del archivo) cuando sea posible."""

    messages = [
        {"role": "system", "content": prompt_sistema},
        {"role": "user", "content": f"Contexto:\n{bloque_contexto}\n\nPregunta: {pregunta}"},
    ]

    # 3. Generar
    prompt_formateado = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    outputs = text_generator(prompt_formateado)
    
    # Extraer solo el texto generado nuevo
    respuesta_completa = outputs[0]["generated_text"]
    respuesta_limpia = respuesta_completa[len(prompt_formateado):]
    
    return respuesta_limpia

# --- 4. BUCLE DE CHAT ---
if __name__ == "__main__":
    print("\n💬 Chat Financiero iniciado. Escribe 'salir' para terminar.")
    while True:
        query = input("\nPregunta: ")
        if query.lower() in ['salir', 'exit']:
            break
        
        print("🔍 Buscando en reportes y analizando...")
        try:
            respuesta = generar_respuesta(query)
            print(f"\n🤖 Respuesta:\n{respuesta}")
        except Exception as e:
            print(f"Error: {e}")