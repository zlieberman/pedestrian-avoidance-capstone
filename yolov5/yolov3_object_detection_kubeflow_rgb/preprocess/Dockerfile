FROM tensorflow/tensorflow:latest

WORKDIR /app

RUN pip install -U scikit-learn numpy opencv-python matplotlib

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN mkdir /app/images/

COPY preprocess.py ./preprocess.py
COPY frames ./frames/
COPY json_data.json ./json_data.json

ENTRYPOINT [ "python", "preprocess.py" ]