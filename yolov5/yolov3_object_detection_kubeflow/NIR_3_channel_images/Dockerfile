FROM ultralytics/yolov3

WORKDIR /app

#RUN apt-get update && apt-get install python3.7 -y && apt-get install python3-pip -y

COPY nir_three_channel.py ./nir_three_channel.py

ADD NIR_3 ./NIR_3

RUN mv NIR_3/* /app

RUN apt-get install dos2unix

RUN dos2unix ./*

RUN ./NIRsetup.sh

RUN pip install smbprotocol

ENTRYPOINT [ "python", "nir_three_channel.py" ]