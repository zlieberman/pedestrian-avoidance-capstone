import glob
import cv2
import re

class Video:

	def __init__(self, image_path='multispectral_imgs/images/*/file2_image*.jpg', video_path='video.mp4'):
		self.image_path = image_path
		self.video_path = video_path

	def atoi(self, text):
		return int(text) if text.isdigit() else text

	def natural_keys(self, text):
		'''
		alist.sort(key=natural_keys) sorts in human order
		http://nedbatchelder.com/blog/200712/human_sorting.html
		(See Toothy's implementation in the comments)
		'''
		return [self.atoi(c) for c in re.split(r'(\d+)', text)]

	def generate_video(self, frames):
		h, w, l = frames[0].shape
		video = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))

		for img in frames:
			video.write(cv2.imread(img))