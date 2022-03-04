import os
import glob
import h5py
import numpy as np
from PIL import Image
import cv2
import re
import random
import shutil
import zipfile
import smbclient
import json
from pathlib import Path
import argparse
# import zipfile
from enum import Enum
from tqdm import tqdm_notebook, tnrange, tqdm

class DataLoader:

	def __init__(self, path, hdf5files=None, json_path=None, labels_path=None, username=None, password=None):
		"""
		Creates a Data Loader object, used for importing training data from the KRI server.
		:param path: Path to the new or existing dataset
		:param hdf5files: Path to HDF5 files to load
		:param json_path: Path to a JSON file either locally or on the KRI server
		:param labels_path: Path to a labels directory or zip file locally.
		:param username: Username to access the KRI file share.
		:param password: Password for the KRI file share.
		"""

		self.chkpt = path + '/chkpt.txt'
		self.path = path
		self.hdf5files = glob.glob(hdf5files)
		self.currfile = None
		self.currindex = 0
		self.fileindex = 0
		self.user = username
		self.passw = password
		self.json_path = json_path
		self.labels_path = labels_path

		if not os.path.isdir(path):
			os.mkdir(path)
		if not os.path.isdir(path + '/images/'):
			os.mkdir(path + '/images/')
			os.mkdir(path + '/images/train/')
			os.mkdir(path + '/images/val/')
			os.mkdir(path + '/images/test/')
		if not os.path.isdir(path + '/labels/'):
			os.mkdir(path + '/labels/')
			os.mkdir(path + '/labels/train/')
			os.mkdir(path + '/labels/val/')
			os.mkdir(path + '/labels/test/')

		self.train_files = glob.glob(self.path + '/images/train/*.jpg')
		self.val_files = glob.glob(self.path + '/images/val/*.jpg')
		self.test_files = glob.glob(self.path + '/images/test/*.jpg')

		print('Already loaded: {} train, {} val,  {} test'.format(len(self.train_files), len(self.val_files),
																  len(self.test_files)))

		lines = []
		if hdf5files is not None and not os.path.isfile(self.chkpt):
			print('creating checkpoint file')
			latest_file = glob.glob(path + '/labels/*/*.txt') if len(glob.glob(path + '/labels/*/*.txt')) > 0 else '0'
			if latest_file != '0':
				latest_file = [int(re.sub('file\w+_image_', '', f.split('/')[3]).strip('.txt')) for f in latest_file]
				file_index = sorted(latest_file)[-1] + 1
			else:
				latest_file = 0
				file_index = 1

			for i, file in enumerate(self.hdf5files):
				f = h5py.File(file, 'r')
				klasses = f['class']
				lines.append(file + ' 0:' + str(len(klasses)) + ' ' + str(i) + '\n')
				file_index += 1
			with open(self.chkpt, 'w+') as f:
				f.writelines(lines)

	def _register_smb(self):
		smbclient.ClientConfig(username=self.user, password=self.passw)
		smbclient.register_session("192.168.152.34", username=self.user, password=self.passw)

	def _save(self, img_name, img, klass, x_center, y_center, width, height):
		with open(self.path + '/labels/train/' + img_name.replace('.jpg', '.txt'), 'w+') as f:
			f.write(' '.join([str(i) for i in [klass, x_center, y_center, width, height]]))

		cv2.imwrite(self.path + '/images/train/' + img_name, img)

	def _to_txt(self, word):
		return word.replace('.jpg', '.txt').replace('.tiff', '.txt')

	def _download_image(self, file_path, resize=(256, 256)):
		with smbclient.open_file(file_path, 'rb') as f:
			arr = np.asarray(bytearray(f.read()), dtype=np.uint8)
			image = cv2.imdecode(arr, -1)
			(h, w) = image.shape[:2]

			image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
		return image, h, w

	def clear_dir(self):
		paths = glob.glob(self.path + '/**/*.*', recursive=True)
		os.remove(self.chkpt)
		for p in paths:
			os.remove(p)

	def save_from_hdf5(self, normalize=True, scale_percent=50):
		if self.hdf5files is None:
			print('no hdf5 files set')
			return

		with open(self.chkpt, 'r+') as chkpt:
			lines = chkpt.readlines()
			done = False
			while not done:
				index = 0
				for i, line in enumerate(lines):
					filedata = line.split(' ')
					print(filedata)
					c = [int(i) for i in filedata[1].split(':')]
					if i >= len(lines) - 1:
						done = True
					if c[1] - c[0] > 1:
						self.currfile = filedata[0]
						self.currindex = c[0] + 1
						self.fileindex = filedata[2]
						index = i
						print('restarting load for {} at {}/{}'.format(filedata[0], c[0], c[1]))
						break

				f = h5py.File(self.currfile, 'r')
				bboxes = f['bbox']
				klasses = f['class']
				datas = f['data']
				times = f['time']

				for idx in range(len(times))[self.currindex:]:
					img_name = 'file' + str(self.fileindex) + '_image_' + str(self.currindex) + '.jpg'
					img_name = img_name.replace('\n', '')
					bbox, klass, data, time = bboxes[idx], klasses[idx], datas[idx], times[idx]
					img = cv2.merge((data[2], data[1], data[0]))
					h, w = data.shape[1:]
					(startX, startY, endX, endY) = bbox[0]
					if normalize:
						x_center = ((endX + startX) / 2) / w
						y_center = ((endY + startY) / 2) / h
						width = (endX - startX) / w
						height = (endY - startY) / h
					else:
						x_center = ((endX + startX) / 2)
						y_center = ((endY + startY) / 2)
						width = (endX - startX)
						height = (endY - startY)
					klass = 0
					imwidth = int(img.shape[1] * scale_percent / 100)
					imheight = int(img.shape[0] * scale_percent / 100)
					dim = (imwidth, imheight)
					img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

					print('saving file {}'.format(img_name), end="\r", flush=True)
					with open(self.path + '/labels/train/' + 'h5_' + img_name.strip('.jpg') + '.txt', 'w+') as f:
						f.write(' '.join([str(i) for i in [klass, x_center, y_center, width, height]]))

					cv2.imwrite(self.path + '/images/train/' + 'h5_' + img_name, img)
					self.currindex += 1

					update = lines[index]
					lines[index] = re.sub(' \w+:', ' ' + str(self.currindex) + ':', update)
					chkpt.seek(0)
					chkpt.writelines(lines)
					chkpt.truncate()
					chkpt.flush()


		print('loaded {} images from hdf5 file'.format(self.currindex))
		self.shuffle(self.path + '/images/train/*.jpg', self.path + '/images/val/*.jpg',
					 self.path + '/images/test/*.jpg')

	def save_from_json(self, normalize=True):
		self._register_smb()
		if self.json_path is None:
			print('No path specified to load from JSON')
			return
		try:
			with smbclient.open_file(self.json_path, 'r') as f:
				json_data = json.load(f)
		except Exception as e:
			try:
				with open(self.json_path, 'r') as f:
					json_data = json.load(f)
			except:
				print('error: unable to find JSON file on KRI server or locally')
				return
        
		jbar = tqdm(json_data)
		for img in jbar:
			self._write_to_YOLO(img, normalize)
			jbar.set_description('Loading image: {}'.format(Path(img['name']).name))
		print('loaded {} images from json file'.format(len(json_data)))
		self.shuffle(self.path + '/images/train/*.jpg', self.path + '/images/val/*.jpg',
					 self.path + '/images/test/*.jpg')

	def save_from_labels(self, normalize=True):
		if self.labels_path is None:
			print('No path to labels specified')
			return

		channel = ['/thermal_images', '/video_color_images']
		if str(self.labels_path).find('thermal') != -1:
			channel = channel[0]
		elif str(self.labels_path).find('color') != -1:
			channel = channel[1]

# 		if not os.path.isdir(self.labels_path):
# 			with zipfile.ZipFile(self.labels_path, 'r') as f:
# 				f.extractall(self.labels_path.strip('.zip'))

		imgs = [Path(x).name for x in glob.glob(self.labels_path + '*.txt')]
		print('{} labels found'.format(len(imgs)))
		self._register_smb()

		train_count = 0
		val_count = 0
		date = re.findall(r'(\d\d\_\d\d\_\d\d)', self.labels_path)[0]
		for root, dirs, files in smbclient.walk(
				r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/' + date + '/Yuneec_Typhoon/', 'r'):
			break
		for di in dirs:
			dirs2 = smbclient.listdir(
				r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/' + date + '/Yuneec_Typhoon/' + di)
			dirs2 = tqdm(dirs2)
			for di2 in dirs2:
				dirs2.set_description('Searching subdirectory: {}'.format(di2))
				for _, _, files in smbclient.walk(
						r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/' + date + '/Yuneec_Typhoon/' + di + '/' + di2 + channel + '/frames/'):
					break
				for file in files:
					if self._to_txt(file) in imgs:
						train_count += 1
						self._write_to_YOLO_from_labels(
							r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/' + date + '/Yuneec_Typhoon/' + di + '/' + di2 + channel + '/frames/' + file,
							self.labels_path, normalize)

		print('loaded {} images from labels file'.format(train_count))
		self.shuffle(self.path + '/images/train/*.jpg', self.path + '/images/val/*.jpg',
					 self.path + '/images/test/*.jpg')

	def _write_to_YOLO(self, img, normalize):
		img_url = img['url']
		img_name = Path(img['name']).name
		base_path = r'\\192.168.152.34/DroneData/MatrixSpaceLabeled/'

		if len(img['labels']) >= 1:
			bounding_box = img['labels'][0]['box2d']
			(startX, startY, endX, endY) = [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'],
											bounding_box['y2']]
			file_path = base_path + img_url[img_url.find(re.findall(r'(\d\d\_\d\d\_\d\d)', img_url)[0]):]
			image, h, w = self._download_image(file_path)

			if normalize:
				x_center = ((endX + startX) / 2) / w
				y_center = ((endY + startY) / 2) / h
				width = (endX - startX) / w
				height = (endY - startY) / h
			else:
				x_center = ((endX + startX) / 2)
				y_center = ((endY + startY) / 2)
				width = (endX - startX)
				height = (endY - startY)

			klass = 0
			self._save('from_json_' + img_name, image, klass, x_center, y_center, width, height)

	def _write_to_YOLO_from_labels(self, img_url, labels, normalize):
		img_name = Path(img_url).name

		image, h, w = self._download_image(img_url)

		with open(labels + self._to_txt(img_name), 'r') as fd:
			data = fd.read()
			try:
				x, y, width, height = [float(i) for i in data.strip('\n').strip(' ').split(' ') if i.isdigit()]
			except:
				return

		if normalize:
			x_center = (x + (width / 2)) / w
			y_center = (y + (height / 2)) / h
			width = width / w
			height = height / h
		else:
			x_center = (x + (width / 2))
			y_center = (y + (height / 2))
			width = width
			height = height

		klass = 0
		self._save('prelabelled_' + img_name, image, klass, x_center, y_center, width, height)

	def write_videos(self, length, video_name, source_images=None):
		if not video_name.endswith('.mp4'):
			video_name = video_name + '.mp4'
		sep = {'h5': [], 'prelabelled': [], 'from_json': []}

		def atoi(text):
			return int(text) if text.isdigit() else text

		def natural_keys(text):
			'''
			alist.sort(key=natural_keys) sorts in human order
			http://nedbatchelder.com/blog/200712/human_sorting.html
			(See Toothy's implementation in the comments)
			'''
			return [atoi(c) for c in re.split(r'(\d+)', Path(text).name)]

		def index_of_substring(l):
			for i, x in enumerate(l):
				if filenum[:-1] + str(int(filenum[-1]) + 1) in x:
					return i

		if source_images is not None:
			files = glob.glob(source_images)
			files.sort(key=natural_keys)
			frame = cv2.imread(files[0])
			h, w, l = frame.shape
			video = cv2.VideoWriter(self.path + '/' + video_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
			vid_length = len(files)
			if length >= vid_length:
				length = vid_length - 1
			start = random.randint(0, vid_length - length)
			stop = start + length
			for f in files[start:stop]:
				video.write(cv2.imread(f))
			print('saved video')
			return
            
            
            
		files = glob.glob(self.path + '/images/*/*.jpg')
        
		for file in files:
			[sep[k].append(file) for k in sep.keys() if k in file]

		for key, fs in sep.items():
			print('saving {} video'.format(key))
			if len(fs) > 0:
				fs.sort(key=natural_keys)
				if key == 'h5':
					filenum = fs[0].split('_')[1]
					frame = cv2.imread(fs[0])
					h, w, l = frame.shape
					video = cv2.VideoWriter(self.path + '/' + key + video_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))

					vid_length = index_of_substring(fs)
					if length >= vid_length:
						length = vid_length - 1
					start = random.randint(0, vid_length - length)
					stop = start + length
					for f in fs[start:stop]:
						if filenum == f.split('_')[1]:
							video.write(cv2.imread(f))
						else:
							break
				else:
					start = random.randint(0, len(fs) - length)
					stop = start + length
					frame = cv2.imread(fs[1])
					h, w, l = frame.shape
					video = cv2.VideoWriter(self.path + '/' + key + video_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
					for f in fs[start:stop]:
						video.write(cv2.imread(f))

	def shuffle(self, train_dir, val_dir, split=0.8):
		current_train_files = glob.glob(train_dir)
		current_val_files = glob.glob(val_dir)
		all_files = current_train_files + current_val_files
		random.shuffle(all_files)

		new_train = all_files[:int(len(all_files) * split)]
		new_val = all_files[int(len(all_files) * split):]

		for t in new_train:
			label = self._to_txt(t).replace('/images/', '/labels/')
			shutil.move(t, t.replace('/val', '/train'))
			shutil.move(label, label.replace('/val', '/train'))

		for t in new_val:
			label = self._to_txt(t).replace('/images/', '/labels/')
			shutil.move(t, t.replace('/train', '/val'))
			shutil.move(label, label.replace('/train', '/val'))

	def shuffle(self, train_dir, val_dir, test_dir, split=(0.6, 0.8)):
		current_train_files = glob.glob(train_dir)
		current_val_files = glob.glob(val_dir)
		current_test_files = glob.glob(test_dir)
		all_files = current_train_files + current_val_files + current_test_files
		random.shuffle(all_files)

		new_train = all_files[:int(len(all_files) * split[0])]
		new_val = all_files[int(len(all_files) * split[0]):int(len(all_files) * split[1])]
		new_test = all_files[int(len(all_files) * split[1]):]
		print('shuffling {} images'.format(len(new_train + new_val + new_test)))

		for t in new_train:
			label = self._to_txt(t).replace('/images/', '/labels/')
			shutil.move(t, t.replace('/val/', '/train/').replace('/test/', '/train/'))
			shutil.move(label, label.replace('/val/', '/train/').replace('/test/', '/train/'))

		for t in new_val:
			label = self._to_txt(t).replace('/images/', '/labels/')
			shutil.move(t, t.replace('/train/', '/val/').replace('/test/', '/val/'))
			shutil.move(label, label.replace('/train/', '/val/').replace('/test/', '/val/'))

		for t in new_test:
			label = self._to_txt(t).replace('/images/', '/labels/')
			shutil.move(t, t.replace('/train/', '/test/').replace('/val/', '/test/'))
			shutil.move(label, label.replace('/train/', '/test/').replace('/val/', '/test/'))


class LoaderType(Enum):
	from_labels = 'labels'
	from_json = 'json'
	from_hdf5 = 'hdf5'
	from_all = 'all'

	def __str__(self):
		return self.value


if __name__ == "__main__":
	PARSER = argparse.ArgumentParser(description='download images')
	PARSER.add_argument('--dataset_path', type=str, help="Path to the new dataset")
	PARSER.add_argument('--hdf5files', type=str, help="Path to the directory containing HDF5 files, in glob format",
						default=None)
	PARSER.add_argument('--json', type=str, help="Path to the JSON file on the KRI server",
						default=None)
	PARSER.add_argument('--labels', type=str, help="Path to the labels file (can be a zip file)",
						default=None)
	PARSER.add_argument('--username', type=str, help="KRI server username", default=None)
	PARSER.add_argument('--password', type=str, help="password", default=None)
	PARSER.add_argument('--type', type=LoaderType, choices=list(LoaderType), default=LoaderType.from_all)

	ARGS = PARSER.parse_args()
	loader = DataLoader(ARGS.dataset_path, hdf5files=ARGS.hdf5files, json_path=ARGS.json, labels_path=ARGS.labels,
						username=ARGS.username, password=ARGS.password)

	if ARGS.type == LoaderType.from_all:
		loader.save_from_labels()
		loader.save_from_json()
		loader.save_from_hdf5()
	elif ARGS.type == LoaderType.from_labels:
		loader.save_from_labels()
	elif ARGS.type == LoaderType.from_json:
		loader.save_from_json()
	elif ARGS.type == LoaderType.from_hdf5:
		loader.save_from_hdf5()