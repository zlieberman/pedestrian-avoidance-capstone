import argparse
import json
import urllib.request as urllib
from pathlib import Path
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os


def _preprocess_data(json_file):
    with open("/app/json_data.json", 'r') as f:
        json_data = json.load(f)

    targets = []
    data = []
    image_names = []
    gt_labels = {}
    for img in json_data:
        img_name = Path(img['name']).name
        filename = "/app/frames/" + img_name

        if len(img['labels']) >= 1:
            bounding_box = img['labels'][0]['box2d']
            (startX, startY, endX, endY) = [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'],
                                            bounding_box['y2']]

            if os.path.isfile(filename):
                image = load_img(filename, target_size=(224, 224))

                w, h = image.size
                startX = float(startX) / w
                startY = float(startY) / h
                endX = float(endX) / w
                endY = float(endY) / h

                # load the image and preprocess it
                image = load_img(filename, target_size=(224, 224))
                image = img_to_array(image)
                data.append(image)
                targets.append((startX, startY, endX, endY))
                image_names.append(filename)
                gt_labels[img_name] = [int(bounding_box['x1']), int(bounding_box['y1']), int(bounding_box['x2']),
                                       int(bounding_box['y2'])]
    data_array = np.array(data, dtype="float32") / 255.0
    targets_array = np.array(targets, dtype="float32")
    split = train_test_split(data_array, targets_array, image_names, test_size=0.10)
    np.save('data_split.npy', split)

if __name__ == '__main__':
    print('Preprocessing data...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file')
    args = parser.parse_args()
    _preprocess_data(args.json_file)
