import naoqi
from naoqi import ALProxy, ALModule, ALBroker
import numpy as np
import cv2, time, sys, StringIO, json, httplib, wave


nao_ip = "192.168.1.137"
nao_port = 9559

global motion_p, posture_p, face_det_p, memory_p, tts_p, speech_rec_p, video_p, wit_speech


class WitAi(ALModule):
	def __init__(self, name):
		try:
			ALModule.__init__(self, name)
			self.ALAudioDevice = ALProxy("ALAudioDevice")
			self.AudioPlayer = ALProxy("ALAudioPlayer")
		except Exception, e:
			print("Error while creating audio proxies:", e)
			sys.exit(0)

		self.saveFile = wave.open('test.wav', 'wb')
		channels = [0, 1, 0, 0]
		self.ALAudioDevice.setClientPreferences(self.getName(), 48000, channels, 0, 0)


	def processRemote(self, inputBuff):
		self.saveFile.write(inputBuff)

	def startWit(self):
		self.headers = {'Authorization':'Bearer 3AHM4NJZN4X5EBLG2AZXTTRG7U67AIU2'}
		self.headers['Content-Type'] = "audio/raw;encoding=signed_integer;bits=16;rate=48000;endian=little"

	def startAudioTest(self, duration=3):
		self.startWit()
		self.ALAudioDevice.subscribe(self.getName())
		self.AudioPlayer.play(self.AudioPlayer.loadFile("/usr/share/naoqi/wav/begin_reco.wav"))
		time.sleep(duration)
		self.ALAudioDevice.unsubscribe(self.getName())
		self.AudioPlayer.play(self.AudioPlayer.loadFile("/usr/share/naoqi/wav/end_reco.wav"))
		self.startUpload(self.saveFile)

	def startUpload(self, datafile):
		conn = httplib.HTTPSConnection("api.wit.ai")
		conn.request("POST", "/speech", datafile.getvalue(), self.headers)
		response = conn.getresponse()
		data = response.read()
		self.reply = data
	 	print data

	def listenNow(self, duration):
		it = 0
		while  it < 3:
			self.startAudioTest(duration)
			response = json.loads(self.reply)
			print("#"*30 + "\n" + response)
			if response["outcomes"][0]["confidence"] > 0.6:
				return response["outcomes"][0]["_text"]
			else:
				tts_p.say("Sorry, I did not understand what you said. Could you please repeat?")
				it += 1
		return None

def are_you_my_mom():
	answer = wit_speech.listenNow(3)
	if "yes" in answer:
		#returning True BREAKS the while loop
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
			tts_p.say("Are you my mom?")
			if are_you_my_mom():
				# BREAKS the while loop
				return False
		return True

def learnFace():
	pass

def center_face():
	val = memory_p.getData("FaceDetected")
	ShapeInfo = val[1][0][0]
	alpha, beta = ShapeInfo[1:2]
	# FIXME - angles should be converted to radians, put at the right order etc
	motion_p.angleInterpolation(["HeadYaw","HeadPitch"], [alpha, beta], [1.5, 1.5], True)
	return False


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
	res = 1, # resolution
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
		cv2.waitKey()
		return None


	cv2.imshow("Detected circles", out_img)
	cv2.waitKey()
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
			wit_speech = WitAi("NaoWit")

		except Exception, e:
			print("Error while creating proxies:")
			print(str(e))
			sys.exit(0)

		motion_p.wakeUp()
		while face_detection():
			# move head around until face is detected
			time.sleep(0.5)
			joint_list = ["HeadYaw", "HeadPitch"]
			angle_list = [list(np.random.uniform(-0.8, 0.8, 1)), list(np.random.uniform(-0.6, 0.6, 1))]
			times = [[1.25],[1.25]]

			# if False: the angles are added to the current position, else they are calculated relative to the origin
			motion_p.angleInterpolation(joint_list, angle_list, times, True)

		# TODO - define these methods
		learnFace()
		if center_face():
			follow_gaze()

		cam = connect_new_cam()
		while True:
			circles = find_circles(cam)


		posture_p.goToPosture("Sit", 0.7)
		motion_p.rest()
		broker.shutdown()
	except Exception, e:
		print("Error", e)
		posture_p.goToPosture("Sit", 0.7)
		motion_p.rest()
		broker.shutdown()
		sys.exit(0)
