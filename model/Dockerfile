FROM python:3.8-slim
WORKDIR /model
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "train_model.py"]
