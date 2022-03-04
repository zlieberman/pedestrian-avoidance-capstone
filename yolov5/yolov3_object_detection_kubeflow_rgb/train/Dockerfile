FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

RUN pip install -U scikit-learn numpy matplotlib

COPY train.py ./train.py

ENTRYPOINT [ "python", "train.py" ]