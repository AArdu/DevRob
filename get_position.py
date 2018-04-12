from naoqi import ALBroker, ALProxy
import time


ip = "192.168.1.137"
port = 9559


motion = ALProxy("ALMotion", ip, port)

while True:
	try:
		chainName = "LArm"
		
		torso = motion.FRAME_TORSO
		robot = motion.FRAME_ROBOT
		world = motion.FRAME_WORLD
		
		useSensor = False
		
		torso_pos = motion.getPosition(chainName, 0, useSensor)
		robot_pos = motion.getPosition(chainName, 2, useSensor)
		world_pos = motion.getPosition(chainName, 1, useSensor)
		
		print("POSITIONS: \n TORSO: {} {} ROBOT {} {} WORLD {}.".format(torso_pos, "\n"*2, robot_pos, "\n"*2, world_pos))
		time.sleep(2)
	except Exception, e:
		print(e)