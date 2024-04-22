FROM python:3.8-slim
ENV PYTHONPATH "${PYTHONPATH}:/app/model"
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
