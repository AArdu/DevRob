from naoqi import ALProxy, ALBroker
import time

ip = "192.168.1.137"

speech_rec_p = ALProxy("ALSpeechRecognition", ip, 9559)
posture = ALProxy("ALRobotPosture", ip, 9559)
motion = ALProxy("ALMotion", ip, 9559)

posture.goToPosture("Stand", 0.5)
time.sleep(1)
posture.goToPosture("Sit", 0.5)

try:
	speech_rec_p.unsubscribe("Test_ASR")
except:
	pass
motion.rest()
broker = ALBroker("broker", "0.0.0.0", 0, ip, 9559)
broker.shutdown()
