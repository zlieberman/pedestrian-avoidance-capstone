import smbclient
import json
import cv2
from pathlib import Path
import numpy as np
import zipfile
import glob
import argparse


def augment_and_save(dataset_folder, labels_folder, image_path, label_path, img_name, image, w, h):
	with open(labels_folder + img_name.replace('.tiff', '.txt'), 'r') as fd:
		data = fd.read()
		x, y, width, height = [float(i) for i in data.strip('\n').strip(' ').split(' ') if i.isdigit()]
		x_center = (x + (width / 2)) / w
		y_center = (y + (height / 2)) / h
		width = width / w
		height = height / h
		klass = 0

	with open(dataset_folder + label_path + img_name.replace('.tiff', '.txt'), 'w+') as f:
		f.write(' '.join([str(i) for i in [klass, x_center, y_center, width, height]]))
	cv2.imwrite(dataset_folder + image_path + img_name, image)

	# augment data - mirror image left-right
	lrimage = np.fliplr(image)
	lr_x_center = 1 - x_center
	with open(dataset_folder + label_path + img_name.replace('.tiff', 'lr.txt'), 'w+') as f:
		f.write(' '.join([str(i) for i in [klass, lr_x_center, y_center, width, height]]))
	cv2.imwrite(dataset_folder + image_path + img_name.replace('.tiff', 'lr.tiff'), lrimage)

	# flip up-down
	udimage = np.flipud(image)
	ud_y_center = 1 - y_center
	with open(dataset_folder + label_path + img_name.replace('.tiff', 'ud.txt'), 'w+') as f:
		f.write(' '.join([str(i) for i in [klass, x_center, ud_y_center, width, height]]))
	cv2.imwrite(dataset_folder + image_path + img_name.replace('.tiff', 'ud.tiff'), udimage)

	# mirror left-right and up-down
	fimage = np.flipud(lrimage)
	with open(dataset_folder + label_path + img_name.replace('.tiff', 'f.txt'), 'w+') as f:
		f.write(' '.join([str(i) for i in [klass, lr_x_center, ud_y_center, width, height]]))
	cv2.imwrite(dataset_folder + image_path + img_name.replace('.tiff', 'f.tiff'), fimage)


def write_to_YOLO(dataset_folder, labels_folder, img_url, val=False, test=False):
	img_name = Path(img_url).name
	channel_urls = [img_url, img_url.replace('NIR1', 'NIR2'), img_url.replace('NIR1', 'NIR3')]
	three_channel = []

	for url in channel_urls:
		with smbclient.open_file(img_url, 'rb') as f:
			arr = np.asarray(bytearray(f.read()), dtype=np.uint8)
			image = cv2.imdecode(arr, -1)
			(h, w) = image.shape[:2]
			image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
			three_channel.append(image)

	image = np.stack(three_channel, axis=2)

	if val:
		augment_and_save(dataset_folder, labels_folder, '/images/val/', '/labels/val/', img_name, image, w, h)

	else:
		augment_and_save(dataset_folder, labels_folder, '/images/train/', '/labels/train/', img_name, image, w, h)


def mount_smb(username, password, smb_paths, train_dataset_folder, val_dataset_folder, train_labels_folder,
			  val_labels_folder, train_imgs, val_imgs):
	# Optional - specify the default credentials to use on the global config object
	smbclient.ClientConfig(username=username, password=password)

	# Optional - register the credentials with a server (overrides ClientConfig for that server)
	smbclient.register_session("192.168.152.34", username=username, password=password)

	train_count = 0
	val_count = 0
	for root, dirs, files in smbclient.walk(
			r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/05_13_20/Yuneec_Typhoon/2020-05-07-20-27-06/', 'r'):
		break
	for di in dirs:
		for _, _, files in smbclient.walk(
				r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/05_13_20/Yuneec_Typhoon/2020-05-07-20-27-06/' + di + smb_path):
			break
		for file in files:
			if file.replace('.tiff', '.txt') in train_imgs:
				train_count += 4
				write_to_YOLO(train_dataset_folder, train_labels_folder,
							  r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/05_13_20/Yuneec_Typhoon/2020-05-07-20-27-06/' + di + smb_path + file)

	for root, dirs, files in smbclient.walk(
			r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/06_03_20/Yuneec_Typhoon/2020-05-07-20-38-19/', 'r'):
		break
	for di in dirs:
		for _, _, files in smbclient.walk(
				r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/06_03_20/Yuneec_Typhoon/2020-05-07-20-38-19/' + di + smb_path):
			break
		for file in files:
			if file.replace('.tiff', '.txt') in val_imgs:
				val_count += 4
				write_to_YOLO(val_dataset_folder, val_labels_folder,
							  r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/06_03_20/Yuneec_Typhoon/2020-05-07-20-38-19/' + di + smb_path + file,
							  val=True)

	print("{} training images".format(train_count))
	print("{} validation images".format(val_count))


if __name__ == "__main__":
	train_files = [Path(p).name for p in glob.glob('labels/labels_05_13_20_multispectral/*.txt')]
	val_files = [Path(p).name for p in glob.glob('labels/labels_06_03_20_multispectral/*.txt')]
	PARSER = argparse.ArgumentParser(description='download images')
	PARSER.add_argument('username', type=str, help="username")
	PARSER.add_argument('password', type=str, help="password")
	ARGS = PARSER.parse_args()
	username = ARGS.username
	password = ARGS.password

	train_files = [Path(p).name for p in glob.glob('labels/labels_05_13_20_multispectral/*.txt')]
	val_files = [Path(p).name for p in glob.glob('labels/labels_06_03_20_multispectral/*.txt')]
	print('training files: {}'.format(len(train_files)))
	print('validation files: {}'.format(len(val_files)))

	smb_path = '/ms_2_images/NIR1/frames/'
	train_dataset_folder = 'yolov3/labels_05_13_20_NIR'
	val_dataset_folder = 'yolov3/labels_06_03_20_NIR'
	train_labels_folder = 'labels/labels_05_13_20_multispectral/'
	val_labels_folder = 'labels/labels_06_03_20_multispectral/'
	mount_smb(username, password, smb_path, train_dataset_folder, val_dataset_folder, train_labels_folder,
			  val_labels_folder, train_files, val_files)