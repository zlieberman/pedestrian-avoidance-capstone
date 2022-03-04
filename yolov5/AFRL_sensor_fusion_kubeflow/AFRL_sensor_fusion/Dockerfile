FROM python:3.8.8

WORKDIR /app

COPY yolov3_deepsort.py ./yolov3_deepsort.py

ADD fusion_files ./fusion_files

RUN mv ./fusion_files/* /app

RUN python --version

RUN python3 --version

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

#RUN apt-get install dos2unix
#RUN dos2unix ./*
RUN pip install --upgrade setuptools
RUN pip install --upgrade -v -r /app/envs/requirements.txt

ENTRYPOINT ["python", "yolov3_deepsort.py"]