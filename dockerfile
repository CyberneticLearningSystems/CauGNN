FROM python:3.8

# Set the working directory
WORKDIR /home/app
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD