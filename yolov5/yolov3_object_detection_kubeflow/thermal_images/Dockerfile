FROM ultralytics/yolov3

WORKDIR /app

#RUN apt-get update && apt-get install python3.7 -y && apt-get install python3-pip -y

COPY thermal.py ./thermal.py

ADD thermal ./thermal

RUN mv thermal/* /app

RUN apt-get install dos2unix

RUN dos2unix ./*

RUN ./yolov3thermalsetup.sh

RUN pip install smbprotocol

ENTRYPOINT [ "python", "thermal.py" ]