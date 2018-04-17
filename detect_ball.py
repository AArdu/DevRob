import cv2
import numpy as np
import time
from naoqi import ALProxy, ALBroker

class DetectBalls:
	def __init__(self, ip, port):
		# self.broker = ALBroker("broker", "0.0.0.0", 0, ip, port)
		self.video_p = ALProxy("ALVideoDevice", ip, port)
		# self.new_cam

	
	def new_cam(self):
		cam_name = "camera"
		cam_type = 0 # top camera
		res = 1 # max resolution
		colspace = 13 # BGR colors
		fps = 10 # frame per second
		
		# unsubscribe all cameras 
		cams = self.video_p.getSubscribers()
		for cam in cams:
			self.video_p.unsubscribe(cam)
			
		# subscribe a new camera
		self.cam = self.video_p.subscribeCamera(cam_name, cam_type, res, colspace, fps)
	
	def get_single_frame(self):
		# get image
		image_container = self.video_p.getImageRemote(self.cam)
		width = image_container[0]
		height = image_container[1]
		
		# pixel data
		values = map(ord, list(image_container[6]))
		# pixel values into numpy array 
		image = np.array(values, np.uint8).reshape((height, width, 3))
		
		cv2.imshow("What I see", image)
		cv2.imwrite("ball.png", image)
		# time.sleep(10)
		cv2.waitKey()
		
		
if __name__ == "__main__":
	nao_ip = "192.168.1.137"
	nao_port = 9559
	db = DetectBalls(nao_ip, nao_port)
	db.new_cam()
	db.get_single_frame()
