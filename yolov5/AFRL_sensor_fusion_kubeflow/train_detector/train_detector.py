import glob
import subprocess
import argparse
import sys
import os

def train(video_color_images):
	train_images = glob.glob(str(video_color_images) + '/images/train/*.jpg')
	val_images = glob.glob(str(video_color_images) + '/images/val/*.jpg')

	with open('train.txt', 'w+') as f:
		for image in sorted(train_images):
			image = image + '\n'
			f.write(image)

	with open('val.txt', 'w+') as f:
		for image in sorted(val_images):
			image = image + '\n'
			f.write(image)

	with open('yolov3.data', 'w+') as data:
		lines = ['classes=1\n',
				 'train=train.txt\n',
				 'valid=val.txt\n',
				 'names=/app/yolov3.names\n',
				 ]
		data.writelines(lines)

	subprocess.call(
		'python /app/yolov3-archive/train.py --data /app/yolov3.data --batch-size 8 --epochs 1 --cfg /app/yolov3-archive/cfg/yolo-drone.cfg --weights '.split(
			' '))
	os.chdir('/app/yolov3-archive')
	subprocess.call([sys.executable, "-c", "from models import *; convert('/app/yolov3-archive/cfg/yolo-drone.cfg', '/app/yolov3-archive/weights/last.pt')"])


if __name__ == "__main__":
	PARSER = argparse.ArgumentParser(description='download images')
	PARSER.add_argument('--images', type=str, help="video color images")
	ARGS = PARSER.parse_args()
	print('training...')

	vci = ARGS.images
	train(vci)
