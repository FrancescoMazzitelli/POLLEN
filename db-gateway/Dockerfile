FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y curl
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
# Copia tutto il contenuto della cartella locale mantenendo i nomi originali
COPY . .

EXPOSE 5000

CMD ["python", "db-gateway.py"]