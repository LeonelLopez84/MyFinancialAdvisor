FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos
COPY requirements.txt .

# Instalar los requerimientos
RUN pip install --no-cache-dir -r requirements.txt

# Instalar bitsandbytes y accelerate explicitly as they are required for 4-bit LLaMA
RUN pip install --no-cache-dir bitsandbytes accelerate

# Copiar el código fuente (se excluirán carpetas según .dockerignore)
COPY . .

# Comando por defecto para iniciar el chat interactivo
CMD ["python", "chat.py"]
