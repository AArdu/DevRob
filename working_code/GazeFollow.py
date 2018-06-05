# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:37:45 2018

@author: Saskia van der Wegen
"""

import os

import chainer
import chainer.functions as F
import chainer.links as L
import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from chainer import Chain, training
from scipy.misc import imresize

# we hebben nodig: data wat de foto is van 1x3x227x227
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

        return fc_0_0, fc_1_0, fc_0_1, fc_0_m1, fc_m1_0

class trainGazeNet():
    def __init__(self, model, batch_size = 128, input_shape = [227, 227], csv_path="./all_data/train_gaze_det/train_annotations.txt"):
        self.model = model
        self.batch_size = batch_size
        self.csv_path = csv_path
        self.input_shape = input_shape
        self.data_csv = pd.read_csv(self.csv_path, header=None)

    def __next__(self):
        return self.next()

    def __call__(self):
        return self

    def augment(self, img, series):
        y_bbox = series[3].values[0]
        y_ann = series[6].values[0]
        y_icen = series[8].values[0]
        aug_series = series.copy()

        # augment data with prob .5
        if np.random.rand() >= 0.5:
            # flip horizontally
            img = img[:,::-1, :]
            # flip bounding box
            y_bbox = 1 - y_bbox
            # flip position of eyes
            y_ann = 1 - y_ann
            # flip gaze direction
            y_icen = 1 - y_icen

        if np.random.rand() >= 0.5:
            # create random crop
            x_start = int(np.random.uniform(0, 0.8) * img.shape[0]) - 1
            x_crop = int(np.random.uniform(0.01, 0.2) * img.shape[0]) - 1
            y_start = int(np.random.uniform(0, 0.8) * img.shape[1]) - 1
            y_crop = int(np.random.uniform(0.01, 0.2) * img.shape[1]) - 1

            img[x_start:x_start + x_crop, y_start:y_start + y_crop, :] = 0

        aug_series[6] = y_ann
        aug_series[8] = y_icen
        aug_series[3] = y_bbox

        return img, aug_series

    def fit_shape_of_inputs(img_resize, eye_image_resize):
        """From:
        https://github.com/pieterwolfert/engagement-l2tor/blob/master/script/gaze_predict.py

        Fits the input for the forward pass.
        """
        input_image_resize = img_resize.reshape([img_resize.shape[0], \
                                                   img_resize.shape[1], \
                                                   img_resize.shape[2], 1])
        input_image_resize = input_image_resize.transpose(3, 2, 0, 1)

        eye_image_resize = eye_image_resize.reshape([eye_image_resize.shape[0], \
                                                    eye_image_resize.shape[1], \
                                                    eye_image_resize.shape[2], 1])
        eye_image_resize = eye_image_resize.transpose(3, 2, 0, 1)
        return input_image_resize, eye_image_resize

    def eyeGrid(img, headlocs):
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

    def preprocess_image(self, in_img, in_series):
        """From:
        https://github.com/pieterwolfert/engagement-l2tor/blob/master/script/gaze_predict.py

        Output images of prepImages are exactly the same as the matlab ones
        Keyword Arguments:
        img --  image with subject for gaze calculation
        """

        img, series = self.augment(in_img, in_series)

        head_pos = [series[6], series[7]] # column, row

        alpha = 0.3
        img_resize = None
        #height, width
        #crop of face (input 2)
        wy = int(alpha * img.shape[0])
        wx = int(alpha * img.shape[1])
        center = [int(head_pos[0]*img.shape[1]), int(head_pos[1]*img.shape[0])]
        y1 = int(center[1]-.5*wy) - 1
        y2 = int(center[1]+.5*wy) - 1
        x1 = int(center[0]-.5*wx) - 1
        x2 = int(center[0]+.5*wx) - 1
        #make crop of face from image
        im_face = img[y1:y2, x1:x2, :]

        #subtract mean from images
        places_mean = sio.loadmat('all_data\\train_gaze_det\\model\\model\\places_mean_resize.mat')
        imagenet_mean = sio.loadmat('all_data\\train_gaze_det\\model\\model\\places_mean_resize.mat')
        places_mean = places_mean['image_mean']
        imagenet_mean = imagenet_mean['image_mean']

        #resize image and subtract mean
        img_resize = imresize(img, self.input_shape, interp='bicubic')
        img_resize = img_resize.astype('float32')
        img_resize = img_resize[:,:,[2,1,0]] - places_mean
        img_resize = np.rot90(np.fliplr(img_resize))

        #resize eye image
        eye_image = imresize(im_face, self.input_shape, interp='bicubic')
        eye_image = eye_image.astype('float32')
        eye_image_resize = eye_image[:,:,[2,1,0]] - imagenet_mean
        eye_image_resize = np.rot90(np.fliplr(eye_image_resize))
        #get everything in the right input format for the network
        img_resize, eye_image_resize = fit_shape_of_inputs(img_resize, eye_image_resize)
        z = eyeGrid(img, [x1, x2, y1, y2])
        z = z.astype('float32')

        img = img.transpose(2, 0, 1)

        i_direction = [int(series[8][i]*w), int(series[9][i]*h)]

        return img, img_resize, eye_image_resize, z, i_direction


    def generate_batch(self):
        data_batch = []
        face_batch = []
        eyes_grid_batch = []
        y_batch = []

        # randomly extract examples to create the
        for i in np.random.choice(len(self.data_csv), self.batch_size, replace = 0):
            data_pd_series = self.data_csv.loc[[i]]

            # load image, change color order, normalize (minmax)
            img_array = np.array(cv2.imread(os.path.join("./all_data/train_gaze_det/", data_pd_series[0].values[0])), dtype=np.float32)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array = ((img_array - np.min(img_array))/(np.max(img_array) - np.min(img_array))).astype(np.float32)

            img_out, face_out, i_grid_out, i_direction = self.preprocess_image(img_array, data_pd_series)

            data_batch.append(img_out)
            face_batch.append(face_out)
            eyes_grid_batch.append(i_grid_out)
            y_batch.append(i_direction)

            data_batch.append(img_aug)
            face_batch.append(face_aug)
            eyes_grid_batch.append(i_grid_aug)
            y_batch.append(i_dir_aug)

        return data_batch, face_batch, eyes_grid_batch, y_batch

    def next(self):
        data, face, eyes_grid, y = self.generate_batch()



if __name__ == "__main__":
    GN = GazeNet()
    train = trainGazeNet(GN)
    a,b,c,s = train.generate_batch()
    print(a)



    #
    # model = GazeNet()
    #
    # # run model
    # fc_0_0, fc_1_0, fc_0_1, fc_0_m1, fc_m1_0 = model(data, face, eyes_grid)
    # # caculate loss
    # # THIS DOES NOT WORK YET and should be more efficient with just putting all f's in one matrix.
    # loss = (F.softmax_cross_entropy(fc_0_0, t)+
    #         F.softmax_cross_entropy(fc_1_0, t)+
    #         F.softmax_cross_entropy(fc_0_1, t)+
    #         F.softmax_cross_entropy(fc_m1_0, t)+
    #         F.softmax_cross_entropy(fc_0_m1, t))/5
    #
    # '''
    # Should use optimizers to optimize and specify lr_mult and decay_mult for weights and biases
    # something like:
    # # 1. Change `update_rule` for specific parameter
    # #    `l1` is `Linear` link, which has parameter `W` and `b`
    # classifier_model.predictor.l1.W.update_rule.hyperparam.lr = 0.01
    #
    # # 2. Change `update_rule` for all parameters (W & b) of one link
    # for param in classifier_model.predictor.l2.params():
    #     param.update_rule.hyperparam.lr = 0.01
    # '''
    #
    # '''
    # Should of course be trained. Backprop is used.
    # But we need correct input and labels.
    # '''
