FROM python:3.12.2

RUN apt-get update && apt-get install -y sqlite3 

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app"]
