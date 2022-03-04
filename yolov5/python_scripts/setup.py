import subprocess
import argparse
import os
"""
VCI
Thermal
NIR1
NIR2
NIR3
NIR
Red
Green
Blue
Multispectral
"""


def setup(dataset_name):
	if not os.path.isdir('../yolov3'):
		os.mkdir('../yolov3')
		subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov3', '../yolov3/'])
	if dataset_name is not None and not os.path.isdir('../' + dataset_name):
		os.mkdir('../' + dataset_name)
		os.mkdir('../' + dataset_name + '/images')
		os.mkdir('../' + dataset_name + '/images/train')
		os.mkdir('../' + dataset_name + '/images/val')
		os.mkdir('../' + dataset_name + '/images/test')
		os.mkdir('../' + dataset_name + '/labels')
		os.mkdir('../' + dataset_name + '/labels/train')
		os.mkdir('../' + dataset_name + '/labels/val')
		os.mkdir('../' + dataset_name + '/labels/test')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='setup yolo')
	parser.add_argument('dataset', type=str, help="dataset name", default=None)
	args = parser.parse_args()
	setup(args.dataset)
