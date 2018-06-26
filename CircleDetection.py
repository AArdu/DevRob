import cv2
import numpy as np
from naoqi import ALBroker, ALProxy

nao_ip = "192.168.1.103"
nao_port = 9559


def circle_detect(img):
    """
    Detect all circles in the image
    """
    cimg = img.copy()
    cv2.imshow('Original', cimg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in img[:, :, 2]]
    cv2.imshow('Contrast enhanced', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', img)
    img = cv2.medianBlur(img, 11)
    cv2.imshow('Blur 1', img)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 25, param1=50, param2=30, minRadius=7, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

video_p = ALProxy("ALVideoDevice", nao_ip, nao_port)
cam = connect_new_cam()
while True:
    img = get_remote_image(cam)
    circle_detect(img)
