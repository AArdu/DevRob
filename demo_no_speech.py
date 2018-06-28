import httplib
import json
import math
import pprint
import random
import StringIO
import subprocess
import sys
import time
import wave

import cv2
import numpy as np

import FaceDetection as FD
import matlab.engine as MATLAB
import naoqi
from GazeFollow import GazeNet as GNet
from naoqi import ALBroker, ALModule, ALProxy

nao_ip = "192.168.1.144"
nao_port = 9559


def find_face(camera):
    """
    Moves the head of the NAO to until it finds a face in its
    visual field, then it centers the detected face.
    """
    tts_p.say("I will now start looking for your face.")
    center_of_face = []
    where_are_you = ["I can't detect your face, you should consider looking straight at me",
                    "This is not a hide and seek demo, please make your face visible",
                    "Where are you?"]
    start_time = time.time()
    while len(center_of_face) == 0:
        center_of_face = detect_face(camera)
        if len(center_of_face) > 0:
            center_face(center_of_face)
        else:
            # when no face is on sight, move the head randomly
            move_head_randomly()
            if int(time.time() - start_time) % 6 == 0:
                tts_p.say(where_are_you[np.random.randint(0, 3)])
            time.sleep(0.5)
    if len(detect_face(camera)) > 0:
        tts_p.say("I see a face.")
    else:
        tts_p.say(where_are_you[np.random.randint(0, 3)])
        find_face(camera)


def detect_face(camera):
    """
    Uses the NAO's camera input to detect the center of
    any faces in its visual field.
    """
    img = get_remote_image(camera)
    fd = FD.FaceDetector(img)
    return fd.detectCenterFaces()


def center_face(center_of_face):
    """
    Moves the head so that the first face
    on the list of detected faces is at the center of the visual field
    """
    center_of_face = [center_of_face[0][0] * 640., center_of_face[0][1] * 480.]
    look_at_gazed(center_of_face)


def move_head_randomly():
    """
    Moves the head randomly within a certain range of angles
    """
    joint_list = ["HeadYaw", "HeadPitch"]
    angle_list = [list(np.random.uniform(-0.8, 0.8, 1)),
                  list(np.random.uniform(-0.3, 0.3, 1))]
    times = [[1.25], [1.25]]
    motion_p.angleInterpolation(joint_list, angle_list, times, True)


def follow_gaze(cam, GazeNet):
    """
    Checks if the face is still visible, then detects the gaze direction,
    and sets NAO's head in motion towards the predicted direction of the gaze.
    If no round object is found in that position, the head returns to the previous
    position to detect the gaze again.
    If the object is detected, then the pointing movement starts.
    """
    center_of_face = []
    while len(center_of_face) == 0:
        img = get_remote_image(cam)

        fd = FD.FaceDetector(img, True)
        center_of_face = fd.detectCenterFaces()
        back_to_face = motion_p.getAngles(["HeadYaw", "HeadPitch"], True)
        if len(center_of_face) > 0:
            tts_p.say("I am now trying to follow your gaze.")

            # get gaze directions
            try:
                gaze_coords = GazeNet.getGaze(center_of_face, img)
            except:
                tts_p.say("I could not predict your gaze")
                print("Error in GazeNet: probably prediction outside image.")
                center_of_face = []

            cv2.circle(img, (gaze_coords[0], gaze_coords[1]), 10, (0, 0, 255))
            cv2.imshow("Image", img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            # use gaze directions to look and point in that direction

            # gaze predicted location
            look_at_gazed(gaze_coords)
            time.sleep(1.5)

            # detect object
            closest_ball = find_closest_object(cam)
            if closest_ball is None:
                motion_p.angleInterpolation(["HeadYaw", "HeadPitch"], back_to_face, [1.0, 1.0], True)
                print("I don't see what you are looking at")
                tts_p.say("I don't see what you are looking at")
            # if detected:
            if closest_ball is not None:
				# look in the direction of the gaze
                look_at_gazed(closest_ball)
				# point at the object if it's there
                point_at_gazed(closest_ball, cam)

                # check if the ball is still there
                checking_ball = find_closest_object(cam)
                if checking_ball is None or (abs(checking_ball[0] - 320) > 40 and abs(checking_ball[1] - 240) > 40):
                    tts_p.say("I pointed at the correct ball!")
                    posture_p.goToPosture("Crouch", 0.5)
                else:
                    tts_p.say(" I pointed at the wrong ball.")
                    posture_p.goToPosture("Crouch", 0.5)
                    motion_p.angleInterpolation(["HeadYaw", "HeadPitch"], back_to_face, [1.0, 1.0], True)





def find_closest_object(cam):
    """
    Finds the circles in the visual field, and selects the one that is
    closest to the center in order to point at it.
    """
    circles = find_circles(cam)
    centers = {}
    print(circles)
    # if circles are detected
    if circles is not None:
        centers['coords'] = circles['centers']
        centers['dist'] = []

        for center in centers['coords']:
            centers['dist'].append(
                np.sqrt((center[0] - 320)**2 + (center[1] - 240)**2))
        # select circle with minimum distance from visual field center
        closest = np.argmin(centers['dist'])
        if closest.dtype == list:
            closest = closest[0]
        closest_ball = centers['coords'][closest]
        return [closest_ball[0], closest_ball[1]]
    else:
        return None


def look_at_gazed(coords):
    """
    Transforms the coordinates of the predicted position into radians in order to
    perform a movement of the head towards that position.
    """
    # center coordinates to the center of the visual field
    coords[0] = coords[0] - 320
    coords[1] = coords[1] - 240

    # change coordinates to radians in the visual field (60.97, 47.64)
    # are the camera angles of the nao in radians
    x_angle = -(coords[0] / 640. * 60.97 * math.pi / 180)
    y_angle = coords[1] / 480. * 47.64 * math.pi / 180

    # bound the angle to the maximum extension of NAO's joints
    current_head_yaw = motion_p.getAngles("HeadYaw", True)[0]
    current_head_pitch = motion_p.getAngles("HeadPitch", True)[0]
    max_yaw = 2.0857 - current_head_yaw
    max_pitch = 0.5149 - current_head_pitch
    min_yaw = -2.0857 - current_head_yaw
    min_pitch = -0.6720 - current_head_pitch
    motion_p.angleInterpolation(["HeadYaw", "HeadPitch"], [max(min_yaw, min(
        max_yaw, x_angle)), max(min_pitch, min(max_pitch, y_angle))], [1.5, 1.5], False)
    time.sleep(1.5)


def point_at_gazed(coords, cam):
    """
    Use RBF to obtain the joint angles needed to point at the object
    detected in the visual field.
    """
    head_pitch, head_yaw = motion_p.getAngles(["HeadPitch", "HeadYaw"], True)
    p_in = [head_yaw - (coords[0] / 640. * 60.97 * math.pi / 180),
            head_pitch + coords[1] / 480. * 47.64 * math.pi / 180]

    # get joint angles from RBF using the matlab engine
    arm_ang = np.array(mat_eng.eval(
        "sim(net, [{},{}].');".format(str(p_in[0]), str(p_in[1]))))


    [elbow, sh_roll, sh_pitch, sh_roll_limits] = choose_arm()
    sh_pitch_limits = [2.0857, -2.0857]

    sh_angles = [max(sh_roll_limits[1], min(sh_roll_limits[0], arm_ang[0][0])), max(sh_pitch_limits[1], min(sh_pitch_limits[0], arm_ang[1][0]))]

    # extend elbow only after the hand is far from the body to avoid collisions
    motion_p.angleInterpolation(
        [sh_roll, sh_pitch], sh_angles, [1., 1.], True)
    motion_p.angleInterpolation([elbow], 0.0349, 0.5, True)
    tts_p.say("Were you looking in the direction I am pointing?")
    tts_p.say("Remove the ball from this position if the direction is correct.")
    time.sleep(3)


def choose_arm():
    """
    Selects the arm to perform the pointing according to the direction
    to which the NAO is looking.
    """
    # wait until the head has reached the final position
    time.sleep(1)
    if (motion_p.getAngles('HeadYaw', False) > 0):
        return ["LElbowRoll", "LShoulderRoll", "LShoulderPitch", [1.3265, -0.3142]]
    elif(motion_p.getAngles('HeadYaw', False) <= 0):
        return ["RElbowRoll", "RShoulderRoll", "RShoulderPitch", [0.3142, -1.3265]]


def connect_new_cam(
        cam_name="TeamPiagetsCam",
        cam_type=0,  # 0 is the upper one
        res=2,  # resolution
        colspace=13,  # BGR
        fps=10  # frames per second
):
    """
    Breaks all previous connections with the webcam and creates a new one
    """
    try:
        cams = video_p.getSubscribers()
        # unsubscribe all cameras
        for cam in cams:
            video_p.unsubscribe(cam)
        # subcscribe new camera
        cam = video_p.subscribeCamera(cam_name, cam_type, res, colspace, fps)
        return cam
    except Exception, e:
        print("Error while subscribing a new camera:", e)


def get_remote_image(cam):
    """
    Acquires an image from the assigned webcam
    """
    image_container = video_p.getImageRemote(cam)
    width = image_container[0]
    height = image_container[1]

    # pixel data
    values = map(ord, list(image_container[6]))
    # pixel values into numpy array
    image = np.array(values, np.uint8).reshape((height, width, 3))
    return image


def get_hough_circle(img):
    """
    Applies circle detection using OpenCV Hough gradient
    """
    circ_dict = {}
    cimg = img.copy()

    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255)
                     for pixel in row] for row in img[:, :, 2]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 11)

    # Hough detection for circles
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 1, 25, param1=50, param2=30, minRadius=7, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circ_dict['centers'] = []
        circ_dict['radii'] = []
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
            # paste center and radius info to dict
            circ_dict['centers'].append((i[0], i[1]))
            circ_dict['radii'].append(i[2])

    else:
        cv2.imshow("Detected circles", cimg)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        return None

    cv2.imshow("Detected circles", cimg)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    return circ_dict


def find_circles(cam):
    """
    Inspect the image captured by the NAO's cam and finds colored circles
    """
    try:
        # get image
        cimg = get_remote_image(cam)
    except Exception, e:
        print("Error while getting remote image:", e)
        print("Attempting new cam connection...")
        cam = connect_new_cam()
        find_circles(cam)
    image = cimg.copy()
    detected_circles = get_hough_circle(image)
    if detected_circles is None or detected_circles['radii'][0] == [0]:
        print("No circles found")
        return None
    print("\nPassing the following detected circles \n{}".format(detected_circles))
    return detected_circles


if __name__ == "__main__":

    try:
        # create proxies
        global motion_p, posture_p, face_det_p, memory_p, tts_p, speech_rec_p, video_p, mat_eng#, ReactToTouch
        motion_p = ALProxy("ALMotion", nao_ip, nao_port)
        posture_p = ALProxy("ALRobotPosture", nao_ip, nao_port)
        face_det_p = ALProxy("ALFaceDetection", nao_ip, nao_port)
        memory_p = ALProxy("ALMemory", nao_ip, nao_port)
        tts_p = ALProxy("ALTextToSpeech", nao_ip, nao_port)
        anim_speech = ALProxy("ALAnimatedSpeech", nao_ip, nao_port)
        speech_rec_p = ALProxy("ALSpeechRecognition", nao_ip, nao_port)
        video_p = ALProxy("ALVideoDevice", nao_ip, nao_port)
        broker = ALBroker("broker", "0.0.0.0", 0, nao_ip, nao_port)
        mat_eng = MATLAB.start_matlab()
    except Exception, e:
        print("Error while creating proxies:")
        print(str(e))
        sys.exit(0)

    # initialize GazeFollow model and load the weights
    GazeNet = GNet()
    GazeNet = GazeNet.loadWeights("all_data/train_GazeFollow/binary_w.npz")

    # start the Matlab engine used in the point_at_gazed function
    mat_eng.load("./all_data/new_rbf_angles.mat", nargout=0)

    posture_p.goToPosture("Crouch", 0.5)

    motion_p.wakeUp()
    cam = connect_new_cam()

    # anim_speech.say("Hello everyone! Welcome to this demonstration developed by Team Piaj   et. " \
    # "Please sit in front of my camera, so that I won't spend too much time searching for your face.", 2)
    # posture_p.goToPosture("Crouch", 0.5)

    try:
        for i in range(5):
            tts_p.say("This is trial number {}".format(str(i + 1)))
            # start the demo with finding the face of the caregiver
            find_face(cam)

            # once the face is found, follow the gaze and point
            follow_gaze(cam, GazeNet)
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        posture_p.goToPosture("Crouch", 0.7)

    # at the end of the demo, go back to a resting position and shitdown the broker
    tts_p.say("The demonstration has come to an end. Thank you for your attention.")
    posture_p.goToPosture("Crouch", 0.7)
    motion_p.rest()
    broker.shutdown()
    sys.exit(0)
