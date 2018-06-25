# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:37:45 2018

@author: Saskia van der Wegen
"""
import os
import time
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.misc import imresize as resize
from matplotlib.patches import Circle
import sys
import chainer
import chainer.functions as F
import chainer.links as L
import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from chainer import Chain, training
from skimage.transform import resize as imresize
import chainer.serializers.npz as npz

# we hebben nodig:  data wat de foto is van 1x3x227x227
#                   data foto van het hoofd: ook 1x3x227x227
#                   eyes-grid van 1x169x1x1


class GazeNet(Chain):
    def __init__(self):
         super(GazeNet, self).__init__()
         with self.init_scope():
            # this are the layers that use the full picture
            self.conv1 = L.Convolution2D(in_channels = None, out_channels=96, ksize=11, stride=4, initialW = chainer.initializers.Normal(0.01), initial_bias = 0)
            self.conv2 = L.Convolution2D(in_channels = None,out_channels = 256, pad = 2, ksize = 5, groups = 2, initialW = chainer.initializers.Normal(0.01), initial_bias=0.1)
            self.conv3 = L.Convolution2D(in_channels = None,out_channels = 384, pad = 1, ksize = 3, initialW = chainer.initializers.Normal(0.01), initial_bias=0)
            self.conv4 = L.Convolution2D(in_channels = None,out_channels = 384, pad = 1, ksize = 3, groups = 2, initialW = chainer.initializers.Normal(0.01), initial_bias=0.1)
            self.conv5 = L.Convolution2D(in_channels = None,out_channels = 256, pad = 1, ksize = 3, groups = 2, initialW = chainer.initializers.Normal(0.01), initial_bias=0.1)
            self.conv5_red = L.Convolution2D(in_channels = None,out_channels = 1, ksize = 1, initialW = chainer.initializers.Normal(0.01), initial_bias=1)
            # now the layers that use the picture of the face
            self.conv1_face = L.Convolution2D(in_channels = None,out_channels = 96, ksize = 11, stride = 4, initialW = chainer.initializers.Normal(0.01), initial_bias=0)
            self.conv2_face = L.Convolution2D(in_channels = None,out_channels = 256, pad = 2, ksize = 5, groups = 2, initialW = chainer.initializers.Normal(0.01), initial_bias=0.1)
            self.conv3_face = L.Convolution2D(in_channels = None,out_channels = 384, pad = 1, ksize = 3, initialW = chainer.initializers.Normal(0.01), initial_bias=0)
            self.conv4_face = L.Convolution2D(in_channels = None,out_channels = 384, pad = 1, ksize = 3, groups = 2, initialW = chainer.initializers.Normal(0.01), initial_bias=0.1)
            self.conv5_face = L.Convolution2D(in_channels = None,out_channels = 256, pad = 1, ksize = 3, groups = 2, initialW = chainer.initializers.Normal(0.01), initial_bias=0.1)
            self.fc6_face = L.Linear(None, out_size =500, initialW = chainer.initializers.Normal(0.01), initial_bias=0.5)
            # other layers
            self.fc7_face = L.Linear(None,out_size =400, initialW = chainer.initializers.Normal(0.01), initial_bias=0.5)
            self.fc8_face = L.Linear(None,out_size =200, initialW = chainer.initializers.Normal(0.01), initial_bias=0.5)
            self.importance_no_sigmoid = L.Linear(None,out_size =169, initialW = chainer.initializers.Normal(0.01), nobias = True)
            self.importance_map = L.Convolution2D(in_channels = None,out_channels = 1, pad = 1, ksize = 3, stride=1, initialW = chainer.initializers.Zero(), initial_bias=0)
            self.fc_0_0 = L.Linear(None,out_size =25, initialW = chainer.initializers.Normal(0.01), initial_bias=0)
            self.fc_1_0 = L.Linear(None,out_size =25, initialW = chainer.initializers.Normal(0.01), initial_bias=0)
            self.fc_m1_0 = L.Linear(None,out_size =25, initialW = chainer.initializers.Normal(0.01), initial_bias=0)
            self.fc_0_1 = L.Linear(None,out_size =25, initialW = chainer.initializers.Normal(0.01), initial_bias=0)
            self.fc_0_m1 = L.Linear(None,out_size =25, initialW = chainer.initializers.Normal(0.01), initial_bias=0)

    def __call__(self, data, face, eyes_grid):
        # the network that uses data as input
        pool1 = F.max_pooling_2d(F.relu(self.conv1(data)), ksize = 3, stride = 2)
        norm1 = F.local_response_normalization(pool1, n = 5, alpha = 0.0001, beta = 0.75)
        pool2 = F.max_pooling_2d(F.relu(self.conv2(norm1)), ksize = 3, stride = 2)
        norm2 = norm1 = F.local_response_normalization(pool2, n = 5, alpha = 0.0001, beta = 0.75)
        conv3 = F.relu(self.conv3(norm2))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))
        conv5_red = F.relu(self.conv5_red(conv5))

        # the network that uses face as input
        pool1_face = F.max_pooling_2d(F.relu(self.conv1_face(face)), ksize = 3, stride = 2)
        norm1_face = F.local_response_normalization(pool1_face, n = 5, alpha = 0.0001, beta = 0.75)
        pool2_face = F.max_pooling_2d(F.relu(self.conv2_face(norm1_face)), ksize = 3, stride=2)
        norm2_face = F.local_response_normalization(pool2_face, n=5, alpha=0.0001, beta= 0.75)
        conv3_face = F.relu(self.conv3_face(norm2_face))
        conv4_face = F.relu(self.conv4_face(conv3_face))
        pool5_face = F.max_pooling_2d(F.relu(self.conv5_face(conv4_face)), ksize=3, stride = 2)
        fc6_face = F.relu(self.fc6_face(pool5_face))

        # now the eyes
        eyes_grid_flat = F.flatten(eyes_grid)
        eyes_grid_mult = 24*eyes_grid_flat
        eyes_grid_reshaped = F.reshape(eyes_grid_mult,(1,eyes_grid_mult.size))  # give it same ndim as fc6

        # now bring everything together
        face_input = F.concat((fc6_face, eyes_grid_reshaped), axis=1)
        fc7_face = F.relu(self.fc7_face(face_input))
        fc8_face = F.relu(self.fc8_face(fc7_face))
        importance_map_reshape = F.reshape(F.sigmoid(self.importance_no_sigmoid(fc8_face)), (1,1,13,13))
        fc_7 = conv5_red * self.importance_map(importance_map_reshape)
        fc_0_0 = self.fc_0_0(fc_7)
        fc_1_0 = self.fc_1_0(fc_7)
        fc_0_1 = self.fc_0_1(fc_7)
        fc_m1_0 = self.fc_m1_0(fc_7)
        fc_0_m1 = self.fc_0_m1(fc_7)

        return {'fc_0_0': fc_0_0, 'fc_1_0': fc_1_0, 'fc_0_1': fc_0_1, 'fc_0_m1': fc_0_m1, 'fc_m1_0': fc_m1_0}

    def loadWeights(self, npz_path):
        npz_f = np.load(npz_path)
        keys = ["arr_" + str(k_id) for k_id in range(len(npz_f.keys()))]

        self.conv1.W.data = npz_f[keys[0]]
        self.conv2.W.data = npz_f[keys[2]]
        self.conv3.W.data = npz_f[keys[4]]
        self.conv4.W.data = npz_f[keys[6]]
        self.conv5.W.data = npz_f[keys[8]]
        self.conv5_red.W.data = npz_f[keys[10]]
        # now the layers that use the picture of the face
        self.conv1_face.W.data = npz_f[keys[12]]
        self.conv2_face.W.data = npz_f[keys[14]]
        self.conv3_face.W.data = npz_f[keys[16]]
        self.conv4_face.W.data = npz_f[keys[18]]
        self.conv5_face.W.data = npz_f[keys[20]]
        self.fc6_face.W.data = npz_f[keys[22]]
        # other layers
        self.fc7_face.W.data = npz_f[keys[24]]
        self.fc8_face.W.data = npz_f[keys[26]]
        self.importance_no_sigmoid.W.data = npz_f[keys[28]]
        self.importance_map.W.data = npz_f[keys[30]]
        self.fc_0_0.W.data = npz_f[keys[32]]
        self.fc_1_0.W.data = npz_f[keys[34]]
        self.fc_m1_0.W.data = npz_f[keys[36]]
        self.fc_0_1.W.data = npz_f[keys[38]]
        self.fc_0_m1.W.data = npz_f[keys[40]]

        self.conv1.b.data = npz_f[keys[0 + 1]]
        self.conv2.b.data = npz_f[keys[2 + 1]]
        self.conv3.b.data = npz_f[keys[4 + 1]]
        self.conv4.b.data = npz_f[keys[6 + 1]]
        self.conv5.b.data = npz_f[keys[8 + 1]]
        self.conv5_red.b.data = npz_f[keys[10 + 1]]
        # now the layers that use the picture of the face
        self.conv1_face.b.data = npz_f[keys[12 + 1]]
        self.conv2_face.b.data = npz_f[keys[14 + 1]]
        self.conv3_face.b.data = npz_f[keys[16 + 1]]
        self.conv4_face.b.data = npz_f[keys[18 + 1]]
        self.conv5_face.b.data = npz_f[keys[20 + 1]]
        self.fc6_face.b.data = npz_f[keys[22 +1]]
        # other layers
        self.fc7_face.b.data = npz_f[keys[24 + 1]]
        self.fc8_face.b.data = npz_f[keys[26 + 1]]
        self.importance_no_sigmoid.b = chainer.variable.Parameter(npz_f[keys[28 + 1]])
        self.importance_map.b.data = npz_f[keys[30 + 1]]
        self.fc_0_0.b.data = npz_f[keys[32 + 1]]
        self.fc_1_0.b.data = npz_f[keys[34 + 1]]
        self.fc_m1_0.b.data = npz_f[keys[36 + 1]]
        self.fc_0_1.b.data = npz_f[keys[38 + 1]]
        self.fc_0_m1.b.data = npz_f[keys[40 + 1]]

        return self

    def prepImages(self, img, e):
        """From:
        https://github.com/pieterwolfert/engagement-l2tor/blob/master/script/gaze_predict.py

        Output images of prepImages are exactly the same as the matlab ones
        Keyword Arguments:
        img --  image with subject for gaze calculation
        e --  head location (relative) [x, y]
        """
        input_shape = [227, 227]
        alpha = 0.3
        #img = imread(img)
        img_resize = None
        # height, width
        # crop of face (input 2)
        print("\nSize of the image is wy: {}, wx{}".format(img.shape[0], img.shape[1]))                                 # Delete this statement
        wy = int(alpha * img.shape[0])
        wx = int(alpha * img.shape[1])
        center = [int(e[0][0] * img.shape[1]), int(e[0][1] * img.shape[0])]
        y1 = int(center[1] - .5 * wy) - 1
        y2 = int(center[1] + .5 * wy) - 1
        x1 = int(center[0] - .5 * wx) - 1
        x2 = int(center[0] + .5 * wx) - 1
        # make crop of face from image
        im_face = img[y1:y2, x1:x2, :]

        # subtract mean from images
        places_mean = sio.loadmat('all_data/places_mean_resize.mat')
        imagenet_mean = sio.loadmat('all_data/imagenet_mean_resize.mat')
        places_mean = places_mean['image_mean']
        imagenet_mean = imagenet_mean['image_mean']

        # resize image and subtract mean
        img_resize = resize(img, input_shape, interp='bicubic')
        img_resize = img_resize.astype('float32')
        img_resize = img_resize[:, :, [2, 1, 0]] - places_mean
        img_resize = np.rot90(np.fliplr(img_resize))

        # resize eye image
        eye_image = resize(im_face, input_shape, interp='bicubic')
        eye_image = eye_image.astype('float32')
        eye_image_resize = eye_image[:, :, [2, 1, 0]] - imagenet_mean
        eye_image_resize = np.rot90(np.fliplr(eye_image_resize))
        # get everything in the right input format for the network
        img_resize, eye_image_resize = self.fit_shape_of_inputs(img_resize, eye_image_resize)
        z = self.eyeGrid(img, [x1, x2, y1, y2])
        z = z.astype('float32')
        return img, img_resize, eye_image_resize, z

    def fit_shape_of_inputs(self, img_resize, eye_image_resize):
        """From:
        https://github.com/pieterwolfert/engagement-l2tor/blob/master/script/gaze_predict.py

        Fits the input for the forward pass."""
        input_image_resize = img_resize.reshape([img_resize.shape[0], \
                                                 img_resize.shape[1], \
                                                 img_resize.shape[2], 1])
        input_image_resize = input_image_resize.transpose(3, 2, 0, 1)

        eye_image_resize = eye_image_resize.reshape([eye_image_resize.shape[0], \
                                                     eye_image_resize.shape[1], \
                                                     eye_image_resize.shape[2], 1])
        eye_image_resize = eye_image_resize.transpose(3, 2, 0, 1)
        return input_image_resize, eye_image_resize

    def eyeGrid(self, img, headlocs):
        """From:
        https://github.com/pieterwolfert/engagement-l2tor/blob/master/script/gaze_predict.py

        Calculates the relative location of the eye.
        Keyword Arguments:
        img -- original image
        headlocs -- relative head location
        """
        w = img.shape[1]
        h = img.shape[0]
        x1_scaled = headlocs[0] / w
        x2_scaled = headlocs[1] / w
        y1_scaled = headlocs[2] / h
        y2_scaled = headlocs[3] / h
        center_x = (x1_scaled + x2_scaled) * 0.5
        center_y = (y1_scaled + y2_scaled) * 0.5
        eye_grid_x = np.floor(center_x * 12).astype('int')
        eye_grid_y = np.floor(center_y * 12).astype('int')
        eyes_grid = np.zeros([13, 13]).astype('int')
        eyes_grid[eye_grid_y, eye_grid_x] = 1
        eyes_grid_flat = eyes_grid.flatten()
        eyes_grid_flat = eyes_grid_flat.reshape(1, len(eyes_grid_flat), 1, 1)
        return eyes_grid_flat

    def predictGaze(self, image, head_image, head_loc):
        return self.__call__(image, head_image, head_loc)

    def postProcessing(self, f_val):
        """From:
        https://github.com/pieterwolfert/engagement-l2tor/blob/master/script/gaze_predict.py
        Combines the 5 outputs into one heatmap and calculates the gaze location
        Keyword arguments:
        f_val -- output of the Caffe model
        """
        fc_0_0 = f_val['fc_0_0'].T
        fc_1_0 = f_val['fc_1_0'].T
        fc_m1_0 = f_val['fc_m1_0'].T
        fc_0_1 = f_val['fc_0_1'].T
        fc_0_m1 = f_val['fc_0_m1'].T
        f_0_0 = F.reshape(fc_0_0, (5, 5))
        f_1_0 = F.reshape(fc_1_0, (5, 5))
        f_m1_0 = F.reshape(fc_m1_0, (5, 5))
        f_0_1 = F.reshape(fc_0_1, (5, 5))
        f_0_m1 = F.reshape(fc_0_m1, (5, 5))
        gaze_grid_list = [self.alpha_exponentiate(f_0_0.data), \
                          self.alpha_exponentiate(f_1_0.data), \
                          self.alpha_exponentiate(f_m1_0.data), \
                          self.alpha_exponentiate(f_0_1.data), \
                          self.alpha_exponentiate(f_0_m1.data)]
        shifted_x = [0, 1, -1, 0, 0]
        shifted_y = [0, 0, 0, -1, 1]
        count_map = np.ones([15, 15])
        average_map = np.zeros([15, 15])
        for delta_x, delta_y, gaze_grids in zip(shifted_x, shifted_y, gaze_grid_list):
            for x in range(0, 5):
                for y in range(0, 5):
                    ix = self.shifted_mapping(x, delta_x, True)
                    iy = self.shifted_mapping(y, delta_y, True)
                    fx = self.shifted_mapping(x, delta_x, False)
                    fy = self.shifted_mapping(y, delta_y, False)
                    average_map[ix:fx + 1, iy:fy + 1] += gaze_grids[x, y]
                    count_map[ix:fx + 1, iy:fy + 1] += 1
        average_map = average_map / count_map
        final_map = resize(average_map, (227, 227), interp='bicubic')
        idx = np.argmax(final_map.flatten())
        [rows, cols] = self.ind2sub2((227, 227), idx)
        y_predict = rows / 227.
        x_predict = cols / 227.
        return final_map, [x_predict, y_predict]

    def alpha_exponentiate(self, x, alpha=0.3):
        """
        From:
        https://github.com/pieterwolfert/engagement-l2tor/blob/master/script/gaze_predict.py
        """
        return np.exp(alpha * x) / np.sum(np.exp(alpha * x.flatten()))

    def ind2sub2(self, array_shape, ind):
        """From:
        https://github.com/pieterwolfert/engagement-l2tor/blob/master/script/gaze_predict.py
        Python implementation of the equivalent matlab method"""
        rows = (ind / array_shape[1])
        cols = (ind % array_shape[1])  # or numpy.mod(ind.astype('int'), array_shape[1])
        return [rows, cols]

    def shifted_mapping(self, x, delta_x, is_topleft_corner):
        """
        From:
        https://github.com/pieterwolfert/engagement-l2tor/blob/master/script/gaze_predict.py
        """
        if is_topleft_corner:
            if x == 0:
                return 0
            ix = 0 + 3 * x - delta_x
            return max(ix, 0)
        else:
            if x == 4:
                return 14
            ix = 3 * (x + 1) - 1 - delta_x
        return min(14, ix)

    def getGaze(self, e, image):
        """From:
        https://github.com/pieterwolfert/engagement-l2tor/blob/master/script/gaze_predict.py
        Calculate the gaze direction in an imageself.
        Keyword arguments:
        e -- list with x,y location of head
        image -- original image
        """
        image, image_resize, head_image, head_loc = self.prepImages(image, e)
        f_val = self.predictGaze(image_resize, head_image, head_loc)
        final_map, predictions = self.postProcessing(f_val)
        print("Predictions = {}".format(predictions))
        print("Shape of the image is: X={}, Y={}".format(np.shape(image)[0], np.shape(image)[1]))
        x = (1-predictions[0]) * np.shape(image)[1]
        y = (1-predictions[1]) * np.shape(image)[0]
        # x = predictions[0] * np.shape(image)[1]
        # y = predictions[1] * np.shape(image)[0]
        x = int(x)
        y = int(y)
        return [x, y]


if __name__ == "__main__":
    GN = GazeNet()
    GN.loadWeights("all_data/train_GazeFollow/binary_w.npz")

