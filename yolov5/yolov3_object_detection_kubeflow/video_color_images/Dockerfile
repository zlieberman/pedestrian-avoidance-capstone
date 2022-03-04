FROM ultralytics/yolov3

WORKDIR /app

#RUN apt-get update && apt-get install python3.7 -y && apt-get install python3-pip -y

COPY video_color_images.py ./video_color_images.py

ADD vci ./video_color_images

RUN mv video_color_images/* /app

RUN apt-get install dos2unix

RUN dos2unix ./*

RUN ./rgbsetup.sh

RUN pip install smbprotocol

ENTRYPOINT [ "python", "video_color_images.py" ]