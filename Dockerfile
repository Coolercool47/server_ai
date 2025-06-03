FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir requirements.txt
EXPOSE 8000
COPY .. .
CMD ["unicorn","main:app", "--host","0.0.0.0", "--port", "8000"]