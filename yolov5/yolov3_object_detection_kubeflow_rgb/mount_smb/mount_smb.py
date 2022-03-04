import smbclient
import argparse
import json
from pathlib import Path
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


def mount_smb(username, password):
    # Optional - specify the default credentials to use on the global config object
    smbclient.ClientConfig(username=username, password=password)
    filepath = r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/05_08_20/DJI_Matice_100/2020-05-08-12-41-04/2020-05-08-12-41-04_0/video_color_images/frames/'

    # Optional - register the credentials with a server (overrides ClientConfig for that server)
    smbclient.register_session("192.168.152.34", username=username, password=password)

    with smbclient.open_file(r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/05_08_20/DJI_Matice_100/2020-05-08-12-41-04-DM-Color_export.json', 'r') as f:
        json_data = json.load(f)

    # Optional - register the credentials with a server (overrides ClientConfig for that server)

    targets = []
    data = []
    image_names = []
    gt_labels = {}
    for img in json_data:
        img_url = img['url']
        img_name = Path(img['name']).name
        filename = "images/" + img_name
        base_path = r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/'

        if len(img['labels']) >= 1:
            bounding_box = img['labels'][0]['box2d']
            (startX, startY, endX, endY) = [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'],
                                            bounding_box['y2']]

            file_path = base_path+img_url[img_url.find('05_08_20'):]
            with smbclient.open_file(file_path, 'rb') as f:
                # with open(img_name, 'wb+') as imagefile:
                #     imagefile.write(f.read())
                arr = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, -1)
                (h, w) = img.shape[:2]
                cv2.imwrite(filename, img)
            # scale the bounding box coordinates relative to the spatial
            # dimensions of the input image
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
    print('mounting file share')
    parser = argparse.ArgumentParser()
    parser.add_argument('--username')
    parser.add_argument('--password')
    args = parser.parse_args()
    mount_smb(args.username, args.password)
