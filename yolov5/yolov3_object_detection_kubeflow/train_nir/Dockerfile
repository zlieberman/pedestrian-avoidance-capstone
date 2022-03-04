FROM ultralytics/yolov3

WORKDIR /app

ADD train_files ./train_files

RUN mv ./train_files/* /app

RUN apt-get install dos2unix
RUN dos2unix ./*

RUN ./NIRsetup.sh

ENTRYPOINT ["python", "train.py"]