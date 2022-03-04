import smbclient
import json
import cv2
from pathlib import Path
import numpy as np
import argparse
from PIL import Image


def save(dataset_folder, image_path, label_path, img_name, image, klass, x_center, y_center, width, height):
	with open(dataset_folder + label_path + img_name.strip('.jpg') + '.txt', 'w+') as f:
		f.write(' '.join([str(i) for i in [klass, x_center, y_center, width, height]]))

	# save the images
	cv2.imwrite(dataset_folder + image_path + img_name, image)


def augment_and_save(dataset_folder, image_path, label_path, img_name, image, klass, x_center, y_center, width, height):
	save(dataset_folder, image_path, label_path, img_name, image, klass, x_center, y_center, width, height)

	# augment data - mirror image left-right
	(h, w) = image.shape[:2]
	lrimage = np.fliplr(image)
	lr_x_center = 1 - x_center
	save(dataset_folder, image_path, label_path, img_name.strip('.jpg') + 'lr.jpg', lrimage, klass, lr_x_center,
		 y_center, width, height)

	# mirror image up-down
	udimage = np.flipud(image)
	ud_y_center = 1 - y_center
	save(dataset_folder, image_path, label_path, img_name.strip('.jpg') + 'ud.jpg', udimage, klass, x_center,
		 ud_y_center, width, height)

	# mirror left-right and up-down
	fimage = np.flipud(lrimage)
	save(dataset_folder, image_path, label_path, img_name.strip('.jpg') + 'f.jpg', fimage, klass, lr_x_center,
		 ud_y_center, width, height)

	# rotate 90 degrees
	rotimage = np.rot90(image)
	rot90_y_center = x_center
	rot90_x_center = 1 - y_center
	save(dataset_folder, image_path, label_path, img_name.strip('.jpg') + 'r90.jpg', rotimage, klass, rot90_x_center,
		 rot90_y_center, width, height)


def mount_smb(username, password, dataset_folder, smb_json_path):
	# Optional - specify the default credentials to use on the global config object
	smbclient.ClientConfig(username=username, password=password)

	# Optional - register the credentials with a server (overrides ClientConfig for that server)
	smbclient.register_session("192.168.152.34", username=username, password=password)

	with smbclient.open_file(smb_json_path, 'r') as f:
		json_data = json.load(f)

	base_path = r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/'

	def write_to_YOLO(img, val=False, test=False):
		img_url = img['url']
		img_name = Path(img['name']).name

		if len(img['labels']) >= 1:
			bounding_box = img['labels'][0]['box2d']
			(startX, startY, endX, endY) = [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'],
											bounding_box['y2']]
			file_path = base_path + img_url[img_url.find('05_08_20'):]
			with smbclient.open_file(file_path, 'rb') as f:
				arr = np.asarray(bytearray(f.read()), dtype=np.uint8)
				image = cv2.imdecode(arr, -1)
				(h, w) = image.shape[:2]

				image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

			x_center = ((endX + startX) / 2) / w
			y_center = ((endY + startY) / 2) / h
			width = (endX - startX) / w
			height = (endY - startY) / h
			klass = 0

			# save the labels
			if val:
				augment_and_save(dataset_folder, '/images/val/', '/labels/val/', img_name, image, klass, x_center,
								 y_center, width, height)

			elif test:
				augment_and_save(dataset_folder, '/images/test/', '/labels/test/', img_name, image, klass, x_center,
								 y_center, width, height)

			else:
				augment_and_save(dataset_folder, '/images/train/', '/labels/train/', img_name, image, klass, x_center,
								 y_center, width, height)

	print('saving training images...')
	images_saved = 0

	for img in json_data[1000:2600]:
		write_to_YOLO(img)
		images_saved += 4

	print('saving validation images...')
	for img in json_data[2600:3050]:
		write_to_YOLO(img, val=True)
		images_saved += 4

	print('images available: {} \n Augmented: {}', images_saved / 4, images_saved)


if __name__ == "__main__":
	PARSER = argparse.ArgumentParser(description='download images')
	PARSER.add_argument('username', type=str, help="username")
	PARSER.add_argument('password', type=str, help="password")
	ARGS = PARSER.parse_args()
	print(ARGS)

	username = ARGS.username
	password = ARGS.password
	mount_smb(username, password, 'yolov3/video_color_images',
			  r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/05_08_20/DJI_Matice_100/2020-05-08-12-41-04-DM-Color_export.json')