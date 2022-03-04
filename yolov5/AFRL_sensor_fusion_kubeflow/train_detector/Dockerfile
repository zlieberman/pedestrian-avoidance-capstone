FROM python:3.8.8

WORKDIR /app

COPY train_detector.py ./train_detector.py

ADD train ./train

RUN mv ./train/* /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get install dos2unix
#RUN dos2unix ./*
RUN pip install --upgrade setuptools
RUN pip install --upgrade -v -r /app/requirements_YOLOv3_archive.txt

ENTRYPOINT ["python", "train_detector.py"]