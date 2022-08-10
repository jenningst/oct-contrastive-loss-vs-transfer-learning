FROM python:3.9.13-slim-buster

# set the working directory and copy files
WORKDIR /api
COPY ./src /api/src
COPY requirements.txt .

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# upgrade pip & install dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# add packages for opencv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# set workdir to src
WORKDIR /api/src

# run the application
EXPOSE 8000
CMD ["python", "-m", "inference"]