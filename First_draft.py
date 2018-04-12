import naoqi
from naoqi import ALProxy, ALModule, ALBroker
import numpy as np
import cv2
import time
import sys

from chainer import Chain, optimizers
import chainer.functions as F
import chainer.links as L


nao_ip = "192.168.1.137"
nao_port = 9559

global motion_p, posture_p, face_det_p, memory_p, tts_p, speech_rec_p


def are_you_my_mom():
	# TODO - make the speech recognition work (get the heard word)
	speech_rec_p.setVocabulary(["yes", "no"], True)
	speech_rec_p.subscribe("Test_ASR")
	time.sleep(3)
	speech_rec_p.unsubscribe("Test_ASR")
	if True: # if "yes" in memory_p.getData("WordRecognized"):
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
			tts_p.say("E?")
			if are_you_my_mom():
				# BREAKS the while loop
				return False 
		return True
	
def center_face():
	pass
	
class Learner(Chain):
	def __init__:
		super(Learner, self).__init__()
        self.fulcon_layer = L.Convolution2D(128, 26*26*self.filter_size)
        self.deconv_layer = L.Convolution2D(in_channels=self.filter_size, out_channels=1, ksize=3, outsize=(28, 28))
        self.batch_norm = L.BatchNormalization(128)
	
def follow_gaze():
	
	pass
	
def get_joint_pos(chainName = "LArm", frame = "robot"):
    if frame = "torso":
		space = 0
	elif frame = "world":
		space = 1
	elif frame = "robot":
		space = 2
    useSensor = False

    # Get the current position of the chainName in the same space
    current = motionProxy.getPosition(chainName, space, useSensor)
	
try:
	try:
		motion_p = ALProxy("ALMotion", nao_ip, nao_port)
		posture_p = ALProxy("ALRobotPosture", nao_ip, nao_port)
		face_det_p = ALProxy("ALFaceDetection", nao_ip, nao_port)
		memory_p = ALProxy("ALMemory", nao_ip, nao_port)
		tts_p = ALProxy("ALTextToSpeech", nao_ip, nao_port)
		speech_rec_p = ALProxy("ALSpeechRecognition", nao_ip, nao_port)
		broker = ALBroker("broker", "0.0.0.0", 0, nao_ip, nao_port)

	except Exception, e:
		print("Error while creating proxies:")
		print(str(e))

	motion_p.wakeUp()
	while face_detection():
		time.sleep(0.5)
		joint_list = ["HeadYaw", "HeadPitch"]
		angle_list = [list(np.random.uniform(-0.8, 0.8, 1)), list(np.random.uniform(-0.6, 0.6, 1))]
		times = [[1.25],[1.25]]

		# if False: the angles are added to the current position, else they are calculated relative to the origin
		motion_p.angleInterpolation(joint_list, angle_list, times, True) 
	
	if center_face():
		follow_gaze()
	
	posture_p.goToPosture("Sit", 0.7)
	motion_p.rest()
	broker.shutdown()
except Exception, e:
	print("Error", str(e))
	posture_p.goToPosture("Sit", 0.7)
	motion_p.rest()
	broker.shutdown()
	sys.exit(0)