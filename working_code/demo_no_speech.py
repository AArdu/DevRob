import naoqi
from naoqi import ALProxy, ALModule, ALBroker
import numpy as np
import cv2, time, sys, StringIO, json, httplib, wave, pprint
import random


nao_ip = "192.168.1.137"
nao_port = 9559

global motion_p, posture_p, face_det_p, memory_p, tts_p, speech_rec_p, video_p


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
        print value
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

def learnFace(face):
	pass

def center_face(face):
	"""
	Moves the head so that the first face
	on the list is at the center of the visual field
	"""
	# pp = pprint.PrettyPrinter(indent=4)
	# print(len(face))
	# pp.pprint(face)
	ShapeInfo = face[1][0][0]
	cameraTorso = face[2]
	alpha = ShapeInfo[1]
	beta = ShapeInfo[2]
	# TODO make it center the biggest (closest) face
	motion_p.angleInterpolation(["HeadYaw","HeadPitch"], [alpha, beta], [1.5, 1.5], False)


def follow_gaze():
	pass


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
	try:
		try:
			# create proxies
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

		motion_p.wakeUp()
		Speecher = SpeechRecognition("Speecher")
		while True:
			faceInfo = face_detection()
			if faceInfo is not None:
				center_face(faceInfo)
				# if areyoumymom(Speecher):
				break
			# move head around until face is detected
			time.sleep(0.5)
			joint_list = ["HeadYaw", "HeadPitch"]
			angle_list = [list(np.random.uniform(-0.8, 0.8, 1)), list(np.random.uniform(-0.6, 0.6, 1))]
			times = [[1.25],[1.25]]

			# if False: the angles are added to the current position, else they are calculated relative to the origin
			motion_p.angleInterpolation(joint_list, angle_list, times, True)


		# TODO - define these methods
		learnFace(faceInfo)
		follow_gaze()

		cam = connect_new_cam()

		while True:
			pointRandom()
			image = get_remote_image(cam)


		circles = find_circles(cam)
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(circles)

		posture_p.goToPosture("Sit", 0.7)
		motion_p.rest()
		broker.shutdown()
	except Exception , e:
		print("Error in __main__", e)
		# posture_p.goToPosture("Sit", 0.7)
		motion_p.rest()
		broker.shutdown()
		sys.exit(0)
