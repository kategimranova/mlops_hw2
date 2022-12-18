FROM python:3.7.4

WORKDIR /app

COPY main.py /app
COPY data_prepare.py /app
COPY requirements.txt /app
COPY models /app/models
COPY data /app/data

# Install dependencies
RUN pip install -r requirements.txt

#EXPOSE 8000
#ENTRYPOINT ["uvicorn"]
CMD ["uvicorn","main:app", "--host", "0.0.0.0", "--port", "8000"]