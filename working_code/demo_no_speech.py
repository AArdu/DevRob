from naoqi import ALProxy, ALModule, ALBroker
import naoqi
import numpy as np
import cv2, time, sys, StringIO, json, httplib, wave, pprint
import random
import subprocess
import FaceDetection as FD
import math
from GazeFollow import GazeNet as GNet
from rbfNew import point


nao_ip = "192.168.1.103"
nao_port = 9559

#motion_p, posture_p, face_det_p, memory_p, tts_p, speech_rec_p, video_p = None


class SpeechRecognition(ALModule):
	def __init__(self, name):
		try:
			p = ALProxy(name)
			p.exit()
		except:
			pass
		ALModule.__init__(self,name)
		self.response = False
		self.value = []
		self.name = name
		self.spr = ALProxy("ALSpeechRecognition")
		self.spr.pause(True)


	def getSpeech(self, wordlist, wordspotting):
		self.response = False
		self.value = []
		self.spr.setVocabulary(wordlist, wordspotting)
		self.spr.pause(False)
		memory_p.subscribeToEvent("WordRecognized", self.name, "onDetect")

	def onDetect(self, keyname, value, subsriber_name):
		self.response = True
		self.value = value
		print(value)
		memory_.unsubscribeToEvent("WordRecognized", self.name)
		self.spr.pause(True)

		if "abort" in self.value[0]:
			self.response = False
			memory_p.unsubscribeToEvent("WordRecognized", Speecher.name)
			sys.exit(0)
			return


def areyoumymom(Speecher):
	tts_p.say("Its'a me, Marvin")
	Speecher.getSpeech(["yes", "no", "abort"], True)
	if "yes" in Speecher.value[0]:
		# Speecher.getSpeech(["yes", "no", "abort"], True)
		memory_p.unsubscribeToEvent("WordRecognized", Speecher.name)
		return True

	return False

def face_detection():
	period = 500
	face_det_p.subscribe("Test_Face", period, 0.0)
	for i in range(0, 5):
		time.sleep(0.5)
		val = memory_p.getData("FaceDetected")
		# Check whether we got a valid output.
		if(val and isinstance(val, list) and len(val) >= 2):
			# a face is detected
			tts_p.say("I see a face")
			face_det_p.unsubscribe("Test_Face")
			return val

	return None


def find_face(camera):
	center_of_face = []
	while len(center_of_face) == 0:
		center_of_face = detect_face(camera)
		if len(center_of_face) > 0:
			center_face(center_of_face)
		else:
			move_head_randomly()
			time.sleep(0.5)
	if len(detect_face(camera)) > 0:
		tts_p.say("I see a face.")
	else:
		tts_p("I thought I found your face but I seem to have lost it.")
		find_face(camera)


def detect_face(camera):
	img = get_remote_image(camera)
	fd = FD.FaceDetector(img, True)
	return fd.detectCenterFaces()


def center_face(center_of_face):
	"""
	Moves the head so that the first face
	on the list is at the center of the visual field
	"""
	center_of_face = [center_of_face[0][0] * 640., center_of_face[0][1] * 480.]
	look_at_gazed(center_of_face)


def move_head_randomly():
	joint_list = ["HeadYaw", "HeadPitch"]
	angle_list = [list(np.random.uniform(-0.8, 0.8, 1)), list(np.random.uniform(-0.6, 0.6, 1))]
	times = [[1.25], [1.25]]
	motion_p.angleInterpolation(joint_list, angle_list, times, True)


def follow_gaze(cam, GazeNet):
	center_of_face = []
	while len(center_of_face) == 0:
		img = get_remote_image(cam)
		fd = FD.FaceDetector(img, True)
		center_of_face = fd.detectCenterFaces()
		if len(center_of_face) > 0:
			# get gaze directions
			gaze_coords = GazeNet.getGaze(center_of_face, img)
			cv2.circle(img, (gaze_coords[0], gaze_coords[1]), 10, (0,0,255))
			cv2.imshow("Image", img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			# use gaze directions to look and point in that direction
			# gaze predicted location
			look_at_gazed(gaze_coords)
			time.sleep(1.5)
			# detect object
			closest_ball = find_closest_object(cam)
			if closest_ball is None:
			#TODO: Let is say it, and maybe look bakc at parent and/or try again
				print("I don't see what you are looking at")
				tts_p.say("I don't see what you are looking at")
			# if detected:
			# TODO: set a maximum distance.
			if closest_ball is not None:
				look_at_gazed(closest_ball)
				point_at_gazed(gaze_coords, cam)
			# gaze object
			# point



def find_closest_object(cam):
	circles = find_circles(cam)
	centers = {}
	centers['coords'] = []
	centers['dist'] = []
	if circles['pink'] is not None:
		for center in circles['pink']['centers']:
			centers['coords'].append(center)
	if circles['blue'] is not None:
		for center in circles['blue']['centers']:
			centers['coords'].append(center)
	if circles['green'] is not None:
		for center in circles['green']['centers']:
			centers['coords'].append(center)
	if circles['yellow'] is not None:
		for center in circles['yellow']['centers']:
			centers['coords'].append(center)

	if len(centers['coords']) > 0:
		for center in centers['coords']:
			centers['dist'].append(np.sqrt((center[0]-320)**2+(center[1]-240)**2))
		closest = np.argmin(centers['dist'])
		if closest.dtype == list:
			closest = closest[0]
		closest_ball = centers['coords'][closest]
		return [closest_ball[0], closest_ball[1]]
	else:
		return None
    

def look_at_gazed(coords):
	coords[0] = coords[0] - 320
	coords[1]= coords[1] - 240

	x_angle = -(coords[0]/640.*60.97*math.pi/180)
	y_angle = coords[1]/480.*47.64*math.pi/180
	current_head_yaw = motion_p.getAngles("HeadYaw", True)[0]
	current_head_pitch = motion_p.getAngles("HeadPitch", True)[0]
	max_yaw = 2.0857 - current_head_yaw
	max_pitch = 0.5149 - current_head_pitch
	min_yaw = -2.0857 - current_head_yaw
	min_pitch = -0.6720 - current_head_pitch
	motion_p.angleInterpolation(["HeadYaw", "HeadPitch"], [max(min_yaw, min(max_yaw, x_angle)), max(min_pitch, min(max_pitch, y_angle))], [1.5, 1.5], False)


def point_at_gazed(coords, cam):
	head_position = motion_p.getPosition('Head', 0, True)
	r_arm_coor = point((coords, head_position), "./all_data/rbfweights.mat")

	# if :
	# 	tts_p.say("I do not understand where you are looking")
	# else:
	# 	# TODO check distinction between head angles and joint angles
	# 	joint_angles = list(joint_angles)
	# 	frame = 0 # motion_p.FRAME_TORSO
	# 	motion_p.positionInterpolations("Head", frame, joint_angles[:5], 7, 1.5)
	# 	arm_chain = choose_arm()
	# 	if find_circles(cam) is not None:
	# 		# FIXME pointing should only be done when the ball is in the expected position, not if any ball is in the visual fiels
	# 		motion_p.positionInterpolations(arm_chain, frame, joint_angles[6:], 7, 1.5)
	# 	else:
	# 		tts_p.say("I do not see the ball that you are looking at")

def sendCoorGetAngl(coords):
	# TODO check that MATLAB and python are writing and reading in the same way
	# write coordinate data to file read by 64 bit environment
	# with open("./all_data/centers_read.txt", 'w+') as fc:
	# 	fc.write(coords)

	counter = 0
	tts_p.say("I'm going to guess where your gaze is pointing")
	while counter < 3:
		time.sleep(1)
		if os.stat("./all_data/centers_read.txt").st_size > 0:
			time.sleep(0.5)
			with open("joints_read.txt", 'r') as fj:
				joint_angles = fj.read()

			# delete content of the file once the joint values are read
			open("centers_read.txt", 'w').close()
			return joint_angles
		else:
			counter += 1

	tts_p.say("I do not know what direction you are looking at")
	return ''


def choose_arm():
	if (motion_p.getAngles('HeadYaw') > 0):
		return 'LArm'
	else:
		return 'RArm'

def get_joint_pos(chainName = "LArm", frame = "robot"):
	if frame == "torso":
		space = 0
	elif frame == "world":
		space = 1
	elif frame == "robot":
		space = 2
	useSensor = False

	# Get the current position of the chainName in the same space
	current = motionProxy.getPosition(chainName, space, useSensor)


def connect_new_cam(
	cam_name = "TeamPiagetsCam",
	cam_type = 0, # 0 is the upper one
	res = 2, # resolution
	colspace = 13, # BGR
	fps = 10 # frames per second
	):
	"""Breaks all previous connections with the webcam andcreates a new one"""
	try:
		cams = video_p.getSubscribers()
		# unsubscribe all cameras
		for cam in cams:
			video_p.unsubscribe(cam)
		# subcscribe new camera
		cam = video_p.subscribeCamera(cam_name, cam_type, res, colspace, fps)
		return cam
	except Exception, e:
		print("Error while subscribing a new camera:" , e)


def get_remote_image(cam):
	"""Acquires an image from the assigned webcam"""
	image_container = video_p.getImageRemote(cam)
	width = image_container[0]
	height = image_container[1]

	# pixel data
	values = map(ord, list(image_container[6]))
	# pixel values into numpy array
	image = np.array(values, np.uint8).reshape((height, width, 3))
	return image


def get_colored_circle(img, l_thr, u_thr):
	"""Applies color thresholds to find circles within that range"""
	out_img = img.copy()
	circ_dict = {}

	hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	c_mask = cv2.inRange(hsv_image, l_thr, u_thr)
	c_image = cv2.bitwise_and(img, img, mask = c_mask)
	kernel = np.ones((9,9), np.uint8)
	opening_c = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel)
	closing_c = cv2.morphologyEx(opening_c, cv2.MORPH_CLOSE, kernel)

	smoothened_mask = cv2.GaussianBlur(closing_c, (9,9), 0)
	c_image = cv2.bitwise_and(img, img, mask = smoothened_mask)
	gray_c = c_image[:,:,2]
	circles = cv2.HoughCircles(gray_c,
		cv2.HOUGH_GRADIENT,		# method of detection
		1,						#
		50,						# minimum distance between circles
		param1 = 50,			#
		param2 = 30,			# accumulator threshold :
		minRadius = 5,			# minimum radius
		maxRadius = 100			# maximum radius
		)

	if circles is not None:
		print("Circles detected!")

		circles = np.round(circles[0, :]).astype("int")
		circ_dict['centers'] =  []
		circ_dict['radii'] = []

		for i in circles:
			# draw he circumference
			cv2.circle(out_img,(i[0],i[1]),i[2],(0,255,0),2)
			# draw center of detected circle
			cv2.circle(out_img,(i[0],i[1]),2,(0,0,255),-1)

			circ_dict['centers'].append((i[0], i[1]))
			circ_dict['radii'].append(i[2])

	else:
		cv2.imshow("Detected circles", out_img)
		cv2.waitKey(1)
		return None


	cv2.imshow("Detected circles", out_img)
	cv2.waitKey(1)
	return circ_dict


def find_circles(cam):
	"""Inspect the image captured by the NAO's cam and finds colored circles"""
	try:
		# get image
		cimg = get_remote_image(cam)

	except Exception, e:
		print("Error while getting remote image:", e)
		print("Attempting new cam connection...")
		cam = connect_new_cam()
		find_circles(cam)

	detected_circles = {}

	image = cimg.copy()


	# threshold for blue
	l_blue = np.array([95, 50, 50])
	u_blue = np.array([115, 255, 255])
	detected_circles['blue'] = get_colored_circle(image, l_blue, u_blue)

	# threshold for green
	l_green = np.array([45, 50, 50])
	u_green = np.array([65, 255, 255])
	detected_circles['green'] = get_colored_circle(image, l_green, u_green)

	# threshold for yellow
	l_yellow = np.array([25, 50, 50])
	u_yellow = np.array([35, 255, 255])
	detected_circles['yellow'] = get_colored_circle(image, l_yellow, u_yellow)

	# threshold for pink
	l_pink = np.array([150, 50, 50])
	u_pink = np.array([180, 255, 255])
	detected_circles['pink'] = get_colored_circle(image, l_pink, u_pink)

	if sum([detected_circles[k] == None for k in detected_circles.keys()]) == 0:
		return None

	return detected_circles

def pointRandom():
	# motionProxy = ALProxy('ALMotion')
	armJoints = [('HeadYaw', -2.0857, 0),
			 ('HeadPitch', -0.330041, 0.200015),
			 ('RShoulderRoll', -1.3265, 0.3142),
			 ('RShoulderPitch', -2.0857, 2.0857)]

	for joint in armJoints:
		angle = random.uniform(joint[1], joint[2])
		motion_p.setAngles(joint[0], angle, 0.1)
		# self.logger.info('Setting {} to {}'.format(joint[0], angle))
		pass


if __name__ == "__main__":
	# try:
		try:
			# create proxies
			global motion_p, posture_p, face_det_p, memory_p, tts_p, speech_rec_p, video_p
			motion_p = ALProxy("ALMotion", nao_ip, nao_port)
			posture_p = ALProxy("ALRobotPosture", nao_ip, nao_port)
			face_det_p = ALProxy("ALFaceDetection", nao_ip, nao_port)
			memory_p = ALProxy("ALMemory", nao_ip, nao_port)
			tts_p = ALProxy("ALTextToSpeech", nao_ip, nao_port)
			speech_rec_p = ALProxy("ALSpeechRecognition", nao_ip, nao_port)
			video_p = ALProxy("ALVideoDevice", nao_ip, nao_port)
			broker = ALBroker("broker", "0.0.0.0", 0, nao_ip, nao_port)
		except Exception, e:
			print("Error while creating proxies:")
			print(str(e))
			sys.exit(0)
		# matlab = subprocess.Popen(["C:\\Users\\univeristy\\Anaconda3\\python.exe", "py64environment.py"])
		GazeNet = GNet()
		GazeNet = GazeNet.loadWeights("all_data/train_GazeFollow/binary_w.npz")

		motion_p.wakeUp()
		cam = connect_new_cam()

		find_face(cam)

		follow_gaze(cam, GazeNet)

		if False:
			circles = find_circles(cam)
			pp = pprint.PrettyPrinter(indent=4)
			pp.pprint(circles)

		posture_p.goToPosture("Crouch", 0.7)
		motion_p.rest()
		broker.shutdown()
	# except Exception , e:
	# 	print("Error in __main__", e)
	# 	# posture_p.goToPosture("Sit", 0.7)
	# 	motion_p.rest()
	# 	broker.shutdown()
	# 	sys.exit(0)
