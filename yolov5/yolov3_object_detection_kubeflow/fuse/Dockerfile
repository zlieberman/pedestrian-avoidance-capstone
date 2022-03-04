FROM ultralytics/yolov3

WORKDIR /app

ADD fuse_files ./fuse_files

RUN mv ./fuse_files/* /app

COPY fuse.py ./fuse.py
#RUN mv ./train_files/* /app

RUN apt-get install dos2unix
RUN dos2unix ./*

RUN ./fusesetup.sh

ENTRYPOINT ["python", "fuse.py"]