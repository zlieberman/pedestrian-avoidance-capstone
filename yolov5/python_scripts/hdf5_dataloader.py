import os
import glob
import h5py
import cv2
import re
import random
import shutil
import argparse


class Hdf5DataLoader():
	def __init__(self, path, files, split=0.8):
		self.chkpt = path + '_chkpt.txt'
		self.path = path
		self.hdf5files = files
		self.currfile = None
		self.currindex = 0
		self.fileindex = 0
		self.split = split

		if not os.path.isdir(path):
			os.mkdir(path)
		if not os.path.isdir(path + '/images/'):
			os.mkdir(path + '/images/')
			os.mkdir(path + '/images/train/')
			os.mkdir(path + '/images/val/')
		if not os.path.isdir(path + '/labels/'):
			os.mkdir(path + '/labels/')
			os.mkdir(path + '/labels/train/')
			os.mkdir(path + '/labels/val/')

		lines = []
		if not os.path.isfile(self.chkpt):
			latest_file = glob.glob(path + '/labels/*/*.txt') if len(glob.glob(path + '/labels/*/*.txt')) > 0 else '0'
			if latest_file != '0':

				latest_file = [int(re.sub('file\w+_image_', '', f.split('/')[3]).strip('.txt')) for f in latest_file]
				file_index = sorted(latest_file)[-1] + 1
			else:
				latest_file = 0
				file_index = 1

			for i, file in enumerate(files):
				f = h5py.File(file)
				klasses = f['class']
				lines.append(file + ' 0:' + str(len(klasses)) + ' ' + str(i) + '\n')
				file_index += 1
			with open(self.chkpt, 'w+') as f:
				f.writelines(lines)

	def saveImages(self):  # path to save; HDF5 files

		with open(self.chkpt, 'r+') as chkpt:
			lines = chkpt.readlines()
			index = 0
			for i, line in enumerate(lines):
				filedata = line.split(' ')
				print(filedata)
				c = [int(i) for i in filedata[1].split(':')]
				if c[1] - c[0] > 1:
					self.currfile = filedata[0]
					self.currindex = c[0] + 1
					self.fileindex = filedata[2]
					index = i
					print('restarting load for {} at {}/{}'.format(filedata[0], c[0], c[1]))
					break

			f = h5py.File(self.currfile)

			bboxes = f['bbox']
			klasses = f['class']
			datas = f['data']
			times = f['time']

			for idx in range(len(times))[self.currindex:]:
				img_name = 'file' + str(self.fileindex) + '_image_' + str(self.currindex) + '.jpg'
				img_name = img_name.replace('\n', '')
				bbox = bboxes[idx]
				klass = klasses[idx]
				data = datas[idx]
				time = times[idx]
				#         data = np.moveaxis(data, [0,1,2], [2, 0, 1])
				img = cv2.merge((data[2], data[1], data[0]))
				h, w = data.shape[1:]
				(startX, startY, endX, endY) = bbox[0]
				x_center = ((endX + startX) / 2) / w
				y_center = ((endY + startY) / 2) / h
				width = (endX - startX) / w
				height = (endY - startY) / h
				klass = 0
				# TODO refactor labels for y_center, width and height -> y_center * 3 / h; width and height * 2 /
				scale_percent = 50  # percent of original size
				imwidth = int(img.shape[1] * scale_percent / 100)
				imheight = int(img.shape[0] * scale_percent / 100)
				dim = (imwidth, imheight)
				img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

				print('saving file to train {}'.format(img_name), end="\r", flush=True)
				with open(self.path + '/labels/train/' + img_name.strip('.jpg') + '.txt', 'w+') as f:
					f.write(' '.join([str(i) for i in [klass, x_center, y_center, width, height]]))

				cv2.imwrite(self.path + '/images/train/' + img_name, img)
				self.currindex += 1

				update = lines[index]
				lines[index] = re.sub(' \w+:', ' ' + str(self.currindex) + ':', update)
				#                 print(lines[index])
				chkpt.seek(0)
				chkpt.writelines(lines)
				chkpt.truncate()
				chkpt.flush()
			self.shuffle(self.path + '/images/train/*.jpg', self.path + '/images/val/*.jpg')

	def shuffle(self, train_dir, val_dir):
		current_train_files = glob.glob(train_dir)
		current_val_files = glob.glob(val_dir)
		all_files = current_train_files + current_val_files
		random.shuffle(all_files)

		new_train = all_files[:int(len(all_files) * self.split)]
		new_val = all_files[int(len(all_files) * self.split):]

		for t in new_train:
			label = t.replace('.jpg', '.txt').replace('images', 'labels')
			shutil.move(t, t.replace('/val', '/train'))
			shutil.move(label, label.replace('/val', '/train'))

		for t in new_val:
			label = t.replace('.jpg', '.txt').replace('images', 'labels')
			shutil.move(t, t.replace('/train', '/val'))
			shutil.move(label, label.replace('/train', '/val'))


if __name__ == "__main__":
	PARSER = argparse.ArgumentParser(description='convert hdf5 files into yolo format')
	PARSER.add_argument('--chkpt', type=str, help="name of the checkpoint file")
	PARSER.add_argument('--filepath', type=str, help="Path to hdf5 files to be used, either hdf5 file or directory")
	PARSER.add_argument('--split', default=None, type=float, help='train_val_split')
	ARGS = PARSER.parse_args()
	chkpt = ARGS.chkpt
	filepath = ARGS.filepath
	# './data_nas1/dronedb/data/*MultiSpectral*processed.hdf5'
	files = None
	if filepath.endswith('.hdf5'):
		files = glob.glob(filepath)
	elif filepath.endswith('/'):
		files = glob.glob(filepath + '*.hdf5')

	loader = Hdf5DataLoader(chkpt, files, 0.8)
	print("{} hdf5 files processed".format(len(files)))
	loader.saveImages()
	loader.shuffle(chkpt + '/images/train/*.jpg', 'multispectral_imgs/images/val/*.jpg')
