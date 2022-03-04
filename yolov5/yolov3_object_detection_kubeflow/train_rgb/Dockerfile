FROM ultralytics/yolov3

WORKDIR /app

ADD train_files ./train_files

RUN mv ./train_files/* /app

RUN apt-get install dos2unix
RUN dos2unix ./*

RUN ./yolov3setup.sh

ENTRYPOINT ["python", "train.py"]

#ENTRYPOINT ["python", "yolov3/train.py", "--img", "256", "--batch", "32", "--epochs",  "20", "--weights", "yolov3.pt", "--data", "video_color_images.yaml",  "--nosave", "--workers", "4"]