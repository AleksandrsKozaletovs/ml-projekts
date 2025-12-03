FROM python:3.12.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY main-hotel.py .

CMD ["streamlit", "run", "app.py", ";", "python", "main-hotel.py"]
