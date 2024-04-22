FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /data

COPY load_data.py .
CMD ["python", "load_data.py"]
