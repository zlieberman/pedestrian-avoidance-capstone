import smbclient
import json
import cv2
from pathlib import Path
import numpy as np
import zipfile
import glob

def write_to_YOLO(dataset_folder, labels_folder, img_url, val=False, test=False):
    img_name = Path(img_url).name

    with smbclient.open_file(img_url, 'rb') as f:
        arr = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)
        (h, w) = image.shape[:2]
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    if val:
        with open(labels_folder + img_name.replace('.tiff', '.txt'), 'r') as fd:
            data = fd.read()
            x,y,width,height = [float(i) for i in data.strip('\n').strip(' ').split(' ') if i.isdigit()]

            x_center = (x + (width / 2 )) / w
            y_center = (y + (height / 2)) / h
            width = width / w
            height = height / h
            klass = 0
            with open(dataset_folder + '/labels/val/' + img_name.replace('.tiff', '.txt'), 'w+') as f:
                f.write(' '.join([str(i) for i in [klass, x_center, y_center, width, height]]))
        # save the images
        cv2.imwrite(dataset_folder + '/images/val/' + img_name, image)

    else:
        with open(labels_folder + img_name.replace('.tiff', '.txt'), 'r') as fd:
            data = fd.read()
            x,y,width,height = [float(i) for i in data.strip('\n').strip(' ').split(' ') if i.isdigit()]

            x_center = (x + (width / 2 )) / w
            y_center = (y + (height / 2)) / h
            width = width / w
            height = height / h
            klass = 0
            with open(dataset_folder + '/labels/train/' + img_name.replace('.tiff', '.txt'), 'w+') as f:
                f.write(' '.join([str(i) for i in [klass, x_center, y_center, width, height]]))
        # save the images
        cv2.imwrite(dataset_folder + '/images/train/' + img_name, image)
        

def mount_smb(username, password, smb_path, dataset_folder, labels_folder, train_imgs, val_imgs):
    # Optional - specify the default credentials to use on the global config object
    smbclient.ClientConfig(username=username, password=password)

    # Optional - register the credentials with a server (overrides ClientConfig for that server)
    smbclient.register_session("192.168.152.34", username=username, password=password)
            
    train_count = 0
    val_count = 0
    for root, dirs, files in smbclient.walk(r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/05_13_20/Yuneec_Typhoon/2020-05-07-20-27-06/', 'r'):
        break
    for di in dirs:
        for _, _, files in smbclient.walk(r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/05_13_20/Yuneec_Typhoon/2020-05-07-20-27-06/' + di + smb_path):
            break
        for file in files:

            if file.replace('.tiff', '.txt') in train_imgs:
                train_count += 1
                write_to_YOLO(dataset_folder, labels_folder, r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/05_13_20/Yuneec_Typhoon/2020-05-07-20-27-06/' + di + smb_path + file)
            elif file.replace('.tiff', '.txt') in val_imgs:
                val_count += 1
                write_to_YOLO(dataset_folder, labels_folder, r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/05_13_20/Yuneec_Typhoon/2020-05-07-20-27-06/' + di + smb_path + file, val=True)
                
    print("{} training images".format(train_count))
    print("{} validation images".format(val_count))
    
    
    
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='download images')
    PARSER.add_argument('username', type=str, help="username")
    PARSER.add_argument('password', type=str, help="password")
    PARSER.add_argument('channel', type=str, help="channel")
    ARGS = PARSER.parse_args()
    paths = {'NIR1': ('/ms_2_images/NIR1/frames/', 'yolov3/labels_05_13_20_NIR1'),
             'NIR2': ('/ms_2_images/NIR2/frames/', 'yolov3/labels_05_13_20_NIR2'),
             'NIR3': ('/ms_2_images/NIR3/frames/', 'yolov3/labels_05_13_20_NIR3'),
             'RED': ('/ms_2_images/Red/frames/', 'yolov3/labels_05_13_20_RED'),
             'GREEN': ('/ms_2_images/Green/frames/', 'yolov3/labels_05_13_20_GREEN'),
             'BLUE': ('/ms_2_images/Blue/frames/', 'yolov3/labels_05_13_20_BLUE')}

    with zipfile.ZipFile('labels.zip', 'r') as zip_ref:
        zip_ref.extractall('labels')
        
    files = glob.glob('labels/labels_05_13_20_multispectral/*.txt')
    smb_path, data_folder = paths[ARGS.channel]
    labels_folder = 'labels/labels_05_13_20_multispectral/'

    username = ARGS.username
    password = ARGS.password
    mount_smb(username, password, smb_path, data_folder, labels_folder, [Path(f).name for f in files[:1600]], [Path(f).name for f in files[1600:]])