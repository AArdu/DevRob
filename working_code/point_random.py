import cv2, time, sys, StringIO, json, httplib, wave, pprint, os, random, json
from naoqi import ALProxy, ALModule, ALBroker
import numpy as np

# Marvin: 137
# Naomi: 103\
#
nao_ip = "169.254.199.32"
nao_port = 9559

global motion_p, posture_p, face_det_p, memory_p, tts_p, speech_rec_p, video_p

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
        cv2.HOUGH_GRADIENT,        # method of detection
        1,                        #
        50,                        # minimum distance between circles
        param1 = 50,            #
        param2 = 30,            # accumulator threshold :
        minRadius = 5,            # minimum radius
        maxRadius = 100            # maximum radius
        )

    if circles is not None:
        print("Circles detected!")

        circles = np.round(circles[0, :]).astype("int")
        circ_dict['centers'] =  []
        # circ_dict['radii'] = []

        for i in circles:
            # draw he circumference
            cv2.circle(out_img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw center of detected circle
            cv2.circle(out_img,(i[0],i[1]),2,(0,0,255),-1)

            circ_dict['centers'].append((i[0], i[1]))
            # circ_dict['radii'].append(i[2])

    else:
        # cv2.imshow("Detected circles", out_img)
        # cv2.waitKey(0)
        return None


    # cv2.imshow("Detected circles", out_img)
    # cv2.waitKey(0)
    return circ_dict

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


def pointRandom():
      # motionProxy = ALProxy('ALMotion')
      armJoints = [('HeadYaw', -1.5708, 0),
               ('HeadPitch', -0.200015, 0.200015),
               ('RShoulderRoll', -1.3265, 0.3142),
               ('RShoulderPitch', -0.785398, 0.785398)]

      angles = []
      arm_pos = []
      head_pos = []
      arm = "RArm"
      head = "Head"
      frame = motion_p.FRAME_TORSO
      useSensorValues = True

      for joint in armJoints:
          angle = random.uniform(joint[1], joint[2])
          angles.append((joint[0], angle))
          motion_p.angleInterpolation([joint[0]], [angle], [2.0], True)# (joint[0], angle, 0.1)
          a = motion_p.getPosition(arm, 0, useSensorValues)
          arm_pos.append(a)
          h = motion_p.getPosition(head, 0, useSensorValues)
          head_pos.append(h)

      return angles, head_pos, arm_pos

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

    # threshold for pink
    l_pink = np.array([150, 50, 50])
    u_pink = np.array([180, 255, 255])
    detected_circles['pink'] = get_colored_circle(image, l_pink, u_pink)

    return detected_circles


if __name__ == "__main__":
    # try:
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
    posture_p.goToPosture("Stand", 0.5)
    cam = connect_new_cam()

    circles_joints = {}
    circles_joints["circles"] = []
    circles_joints["joints"] = []
    circles_joints["head_pos"] = []
    circles_joints["arm_pos"] = []
                
    motion_p.setAngles(["RElbowRoll", "RWristYaw"], [0.0349, -1.8238], 0.3)
    i = 0
    while i < 225:
        angles, head_pos, arm_pos = pointRandom()
        circles = find_circles(cam)
        if circles['pink'] is not None:
            if len(circles['pink']['centers']) == 1:
                print("Pink ball #{} detected".format(i))
                # image = get_remote_image(cam)
                # cv2.imwrite("./arm_images/pointing_{}.png".format(i), image)
                circles_joints["circles"].append(circles['pink']['centers'])
                circles_joints["joints"].append(angles)
                circles_joints["head_pos"].append(head_pos)
                circles_joints["arm_pos"].append(arm_pos)
                

                i += 1
        if i % 8  == 0 and i != 0:
            with open('./data.json', 'w+') as fp:
                json.dump(circles_joints, fp)
            print("I'm going to rest for a bit.")
            motion_p.rest()
            i += 1
            time.sleep(200)
            motion_p.wakeUp()


    print circles_joints

    posture_p.goToPosture("Sit", 0.3)
    motion_p.rest()
    broker.shutdown()
    sys.exit(0)

    # except Exception , e:
    #     print("Error in __main__", e)
    #     posture_p.goToPosture("Stand", 0.7)
    #     motion_p.rest()
    #     broker.shutdown()
    #     sys.exit(0)
