FROM python:3.8

# Create a folder /app is the current working directory
WORKDIR /app

# Copy necessary files to app


COPY ./requirements.txt /app

# Port will be exposed
EXPOSE 4001

# Install necessary libraries
RUN pip install -r requirements.txt --no-cache-dir

COPY ./main.py /app

COPY ./models /app/models

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4001", "--reload"]