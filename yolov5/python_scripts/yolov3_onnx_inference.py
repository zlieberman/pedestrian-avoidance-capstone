import onnxruntime as rt
import cv2
import math
import glob
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import argparse
from videowriter import Video


class PostProcessor:
	def __init__(self, impath=random.choice(
		glob.glob('/home/jovyan/workspace/yolov3/labels_05_13_20_thermal/images/*/*.jpg')),
				 wtpath='/home/jovyan/workspace/YoloThermal.onnx', scale=None):
		image = Image.open(impath)
		if scale is not None:
			resize = transforms.Resize(scale)
			self.image = resize(image)
		else:
			self.image = image
			scale = image.size

		self.bbox = impath.replace('images/', 'labels/').replace('.jpg', '.txt')
		self.impath = impath
		self.w = scale[0]
		self.h = scale[1]
		self.remap = {0: 4, 1: 10, 2: 16}
		self.confindices = [4, 10, 16]
		self.class_confindices = [5, 11, 17]

		self.sess = rt.InferenceSession(wtpath)
		self.input_name = self.sess.get_inputs()[0].name
		self.label_names = [o.name for o in self.sess.get_outputs()]

	def bb_intersection_over_union(self, boxA, boxB):
		# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
		# compute the area of intersection rectangle
		interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
		# return the intersection over union value
		return iou

	def load_bbox(self):
		with open(self.bbox) as f:
			line = f.read()
			line = [float(x) for x in line.split(' ')]
			klass, x, y, w, h = line
			_x1 = (x - w / 2) * self.w
			_x2 = (x + w / 2) * self.w
			_y1 = (y - h / 2) * self.h
			_y2 = (y + h / 2) * self.h
		self.bbox = [(int(_x1), int(_y1)), (int(_x2), int(_y2))]

	def letterbox_image(self, image, size):
		'''resize image with unchanged aspect ratio using padding'''
		iw, ih = image.size
		w, h = size
		scale = min(w / iw, h / ih)
		nw = int(iw * scale)
		nh = int(ih * scale)

		image = image.resize((nw, nh), Image.BICUBIC)
		new_image = Image.new('RGB', size, (128, 128, 128))
		new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
		return new_image

	def preprocess(self, img):
		model_image_size = (227, 227)
		boxed_image = self.letterbox_image(img, tuple(reversed(model_image_size)))
		image_data = np.array(boxed_image, dtype='float32')
		image_data /= 255.
		image_data = np.transpose(image_data, [2, 0, 1])
		image_data = np.expand_dims(image_data, 0)
		return image_data

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def softmax(self, x):
		"""Compute softmax values for each sets of scores in x."""
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	def display_output(self, impath):

		fig = plt.figure(figsize=(16., 16.))
		grid = ImageGrid(fig, 111,  # similar to subplot(111)
						 nrows_ncols=(4, 4),  # creates 4x4 grid of axes
						 axes_pad=0.4,
						 )

		for ax in grid:
			# Iterating over the grid returns the Axes.
			img, title = self.run(random.choice(glob.glob(impath)))
			ax.set_title(title)
			ax.imshow(img)

	def run(self, impath, show=False):
		image = Image.open(impath)
		# input
		image_data = self.preprocess(image)
		image_size = np.array([image.size[1], image.size[0]], dtype=np.int32).reshape(1, 2)
		pred_onx = self.sess.run(self.label_names, {self.input_name: image_data})
		x1 = pred_onx[0][0]
		x2 = pred_onx[1][0]

		obj_confs1 = x1[self.confindices]
		obj_confs2 = x2[self.confindices]
		class_confs1 = x1[self.class_confindices]
		class_confs2 = x2[self.class_confindices]
		confs1 = np.multiply(self.softmax(obj_confs1), self.softmax(class_confs1))
		confs2 = np.multiply(self.softmax(obj_confs2), self.softmax(class_confs2))

		if np.max(confs1) > np.max(confs2):
			x = x1
			confs = confs1
			res = 'low'
			anchors = [[41, 53], [31, 38], [26, 31]]

		else:
			x = x2
			confs = confs2
			res = 'high'
			anchors = [[20, 24], [16, 17], [13, 8]]

		scale = self.w / x.shape[1]
		padding = scale / 2

		m = np.array(np.where(confs == np.max(confs))).reshape(3, )

		try:
			t_x, t_y, t_w, t_h, _, _, = x[self.remap[m[0]] - 4:self.remap[m[0]] + 2, m[1], m[2]]

			anchors = anchors[m[0]]
			b_x = self.sigmoid(t_x) + m[2] * scale + padding
			b_y = self.sigmoid(t_y) + m[1] * scale + padding
			b_w = anchors[1] * np.exp(t_w)
			b_h = anchors[0] * np.exp(t_h)
			x1 = int(b_x - (b_w / 2))
			x2 = int(b_x + (b_w / 2))
			y1 = int(b_y - (b_h / 2))
			y2 = int(b_y + (b_h / 2))
			image = cv2.imread(impath)

			image = np.array(image)
			dim = (self.w, self.h)
			image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
			self.load_bbox()
			IoU = self.bb_intersection_over_union([self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1]],
												  [x1, y1, x2, y2])
			img = cv2.rectangle(image, self.bbox[0], self.bbox[1], color=(255, 0, 0), thickness=1)
			if IoU > 0.0001:
				img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
			if show:
				from matplotlib import pyplot as plt
				plt.imshow(img)
				plt.title('conf: {}'.format(round(np.max(confs)), 4))
				plt.show()
			else:
				return (img, str('conf: {} IoU: {}'.format(round(float(np.max(confs)), 4), round(IoU, 4))))
		except Exception as e:
			return (image, str('conf: 0'))


if __name__ == "__main__":
	PARSER = argparse.ArgumentParser(description='download images')
	PARSER.add_argument('--image_path', type=str, help="Path to the images to run inference on", default=None)
	PARSER.add_argument('--weights_path', type=str, help="Path to the YOLOv3 onnx weights", default=None)
	ARGS = PARSER.parse_args()
	pp = PostProcessor(ARGS.image_path, ARGS.weights_path)
