import cv2
import GazeFollow
import numpy as np


class FaceDetector:
	"""
	Class to detect faces in picture.
	"""
	def __init__(self, img="face.jpg", debug=False):
		self.face_cascade = cv2.CascadeClassifier('./all_data/haarcascade_frontalface_default.xml')
		self.profile_face_cascade = cv2.CascadeClassifier('./all_data/haarcascade_profileface.xml')
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
		profile_faces = self.profile_face_cascade.detectMultiScale(self.gray, 1.3, 5)
		return faces, profile_faces

	def drawFaceBoxes(self):
		faces, prof_faces = self.detectFaces()
		for (x, y, w, h) in faces:
			cv2.rectangle(self.img, (x, y), (x+w, y+h), (255, 0, 0), 2)
		for (x, y, w, h) in prof_faces:
			cv2.rectangle(self.img, (x, y), (x+w, y+h), (255, 0, 0), 2)


	def detectCenterFaces(self):
		centers = []
		height, width, channels = self.img.shape

		faces, prof_faces = self.detectFaces()
		for (x, y, w, h) in faces:
			centers.append([(x + float(w)/2)/width, (y + float(h)/2)/height])
		for (x, y, w, h) in prof_faces:
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

		faces, prof_faces = self.detectFaces()
		eyes = prof_eyes = []
		for (x, y, w, h) in faces:
			roi_gray = self.gray[y:y+h, x:x+w]
			roi_color = self.img[y:y+h, x:x+w]
			eyes = self.eye_cascade.detectMultiScale(roi_gray)

		for (x, y, w, h) in prof_faces:
			roi_gray = self.gray[y:y+h, x:x+w]
			roi_color = self.img[y:y+h, x:x+w]
			prof_eyes = self.eye_cascade.detectMultiScale(roi_gray)

		return eyes, prof_eyes

	def drawEyeBoxes(self):
		faces, prof_faces = self.detectFaces()
		eyes, prof_eyes = self.detectEyes()

		for (x, y, w, h) in faces:
			for (ex, ey, ew, eh) in eyes:
				roi_color = self.img[y:y+h, x:x+w]
				cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
		for (x, y, w, h) in prof_faces:
			for (ex, ey, ew, eh) in prof_eyes:
				roi_color = self.img[y:y+h, x:x+w]
				cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

	def loadIm(self, img):
		if isinstance(img, str):
			self.img = cv2.imread(img)
		else:
			self.img = img
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

	def showIm(self):
		cv2.namedWindow('img', 0)
		cv2.resizeWindow('img', 500, 500)
		cv2.imshow('img', self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == "__main__":
	img = "./all_data/Gaze.jpg"
	fd = FaceDetector(img, True)
	e = fd.detectCenterFaces()
	GN = GazeFollow.GazeNet()
	GN.loadWeights("./all_data/train_GazeFollow/binary_w.npz")
	x, y = GN.getGaze(e[0], img)
	image = cv2.imread(img)
	cv2.line(image, (x - 5, y), (x + 5, y), (255, 0, 0))
	cv2.line(image, (x, y - 5), (x, y + 5), (255, 0, 0))
	cv2.imshow('img', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
