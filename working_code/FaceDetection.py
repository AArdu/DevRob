import cv2
import GazeFollow
from sklearn.model_selection import train_test_split
import os
import scipy.io

class FaceDetector:
	"""
	Class to detect faces in picture.
	"""
	def __init__(self, img="face.jpg", debug=False):
		self.face_cascade = cv2.CascadeClassifier('./all_data/haarcascade_frontalface_default.xml')
		self.eye_cascade = cv2.CascadeClassifier('./all_data/haarcascade_eye.xml')
		self.img = None
		self.gray = None
		self.loadIm(img)
		if debug:
			self.drawFaceBoxes()
			self.drawCenterFaces()
			self.drawEyeBoxes()
			self.showIm()

	def detectFaces(self):
		"""
		Detects faces in an image.
		Input:
		img = cv.imread image Image in which you want to detect faces.
		Output:
		faces = [(topLeft_x, topLeft_y, width, height)] List of quadrupples that describe a box around the faces that were detected.
		"""
		faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
		return faces

	def drawFaceBoxes(self):
		for (x, y, w, h) in self.detectFaces():
			cv2.rectangle(self.img, (x, y), (x+w, y+h), (255, 0, 0), 2)

	def detectCenterFaces(self):
		centers = []
		height, width, channels = self.img.shape
		for (x, y, w, h) in self.detectFaces():
			centers.append([(x + float(w)/2)/width, (y + float(h)/2)/height])
		return centers

	def drawCenterFaces(self):
		height, width, channels = self.img.shape
		for (x, y) in self.detectCenterFaces():
			cv2.line(self.img, (int(x*width)-5, int(y*height)), (int(x*width)+5, int(y*height)), (255,0,0))
			cv2.line(self.img, (int(x*width), int(y*height)-5), (int(x*width), int(y*height)+5), (255,0,0))

	def detectEyes(self):
		"""
		Detects eyes in an image.
		Input:
		img = cv.imread image Image in which you want to detect faces.
		Output:
		eyes = [(topLeft_x, topLeft_y, width, height)] List of quadrupples that describe a box around the eyes that were detected.
		"""
		for (x, y, w, h) in self.detectFaces():
			roi_gray = self.gray[y:y+h, x:x+w]
			roi_color = self.img[y:y+h, x:x+w]
			eyes = self.eye_cascade.detectMultiScale(roi_gray)
		return eyes

	def drawEyeBoxes(self):
		for (x, y, w, h) in self.detectFaces():
			for (ex, ey, ew, eh) in self.detectEyes():
				roi_color = self.img[y:y+h, x:x+w]
				cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

	def loadIm(self, img):
		if isinstance(img, str):
			self.img = cv2.imread(img)
		else:
			self.img = img
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

	def showIm(self):
		cv2.imshow('img', self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == "__main__":
	GN = GazeFollow.GazeNet()
	GN.loadWeights("./all_data/train_GazeFollow/binary_w.npz")
	data = scipy.io.loadmat("C:/Users/The Mountain/Downloads/data/test_annotations.mat")
	f = open("C:/Users/The Mountain/Downloads/data/test_annotations.txt")
	while True:
		data = f.readline()
		data = data.split(",")
		x, y, w, h = [float(i) for i in data[2:6]]
		ex, ey =  [float(i) for i in data[6:8]]
		px, py =  [float(i) for i in data[8:10]]
		img = os.path.join("C:\Users\The Mountain\Downloads\data", data[0])
		image = cv2.imread(img)
		fd = FaceDetector(img)
		e = fd.detectCenterFaces()
		if len(e) > 0:
			cv2.rectangle(image, (int(x*image.shape[1]),int(y*image.shape[0])), (int(x*image.shape[1]+w*image.shape[1]), int(y*image.shape[0]+h*image.shape[0])), (255,255,255))
			cv2.circle(image, (int(ex*image.shape[1]), int(ey*image.shape[0])), 10, (255,255,255))
			cv2.circle(image, (int(px*image.shape[1]), int(py*image.shape[0])), 10, (0,255,255))
			x, y = GN.getGaze(e[0], img)
			print("X {} and Y {}".format(x, y))
			cv2.circle(image,(x, y), 10, (0,255,0))
			cv2.line(image, (x - 5, y), (x + 5, y), (255, 0, 0))
			cv2.line(image, (x, y - 5), (x, y + 5), (255, 0, 0))
			cv2.imshow('img', image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	# print(data["test_bbox"][0][0])
	# x, y, w, h = data["test_bbox"][0][0][0]
	# ex, ey = data["test_eyes"][0][0][0]
	# px, py = data["test_gaze"][0][0][0]
	# img = os.path.join("C:\Users\The Mountain\Downloads\data", data["test_path"][0][0][0])
	# image = cv2.imread(img)
	# cv2.rectangle(image, (int(x*image.shape[1]),int(y*image.shape[0])), (int(x*image.shape[1]+w*image.shape[1]), int(y*image.shape[0]+h*image.shape[0])), (255,255,255))
	# cv2.circle(image, (int(ex*image.shape[1]), int(ey*image.shape[0])), 10, (255,255,255))
	# cv2.circle(image, (int(px*image.shape[1]), int(py*image.shape[0])), 10, (0,255,255))
	# cv2.imshow('img', image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()




	#
	# fc = []
	# for bbox in data["test_bbox"]:
	# 	fc.append([bbox])
	# print(data.keys())




	#
	#
	# for i in range(9):
	# 	img = "./all_data/Gaze" + str(i) + ".jpg"
	# 	fd = FaceDetector(img, True)
	# 	e = fd.detectCenterFaces()
	# 	GN = GazeFollow.GazeNet()
	# 	GN.loadWeights("./all_data/train_GazeFollow/binary_w.npz")
	# 	x, y = GN.getGaze(e[0], img)
	# 	print("X {} and Y {}".format(x, y))
	# 	image = cv2.imread(img)
	# 	cv2.circle(image,(x, y), 10, (0,255,0))
	# 	cv2.line(image, (x - 5, y), (x + 5, y), (255, 0, 0))
	# 	cv2.line(image, (x, y - 5), (x, y + 5), (255, 0, 0))
	# 	cv2.imshow('img', image)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

