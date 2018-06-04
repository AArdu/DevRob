import chainer
from chainer import Chain
from chainer import functions as F
from chainer import links as L
from chainer.dataset import download
from chainer.dataset.convert import concat_examples
from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.functions.array.reshape import reshape
from chainer.functions.math.average import average
from chainer.functions.noise.dropout import dropout
from chainer.functions.normalization.local_response_normalization import \
    local_response_normalization
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from chainer.initializers import constant, uniform
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.inception import Inception
from chainer.links.connection.linear import Linear
from chainer.serializers import npz
from chainer.utils import argument, imgproc
from chainer.variable import Variable


class GazeFollow(Chain):
    def __init__(self):
        super(GazeFollow, self).__init__()
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

    def _make_npz(caffe_path = "./all_data/train_gaze_det/model/model/binary_w.caffemodel",
                     npz_path = "./all_data/train_gaze_det/model/model/GazeFollow.npz"):
        GazeFollow.convert_caffemodel_to_npz(caffe_path, npz_path)
        npz.load_npz(npz_path, self)
        return self

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.
        Args:
        path_caffemodel (str): Path of the pre-trained caffemodel.
        path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = cls(pretrained_model=None)
        _transfer_GazeFollow(caffemodel, chainermodel)
        npz.save_npz(path_npz, chainermodel, compression=False)

    def _transfer_GazeFollow(src, dst):
        # the network that uses data as input
        dst.conv1.W.data = src['conv1'].W.data
        dst.conv2.W.data = src['conv2'].W.data
        dst.conv3.W.data = src['conv3'].W.data
        dst.conv4.W.data = src['conv4'].W.data
        dst.conv5.W.data = src['conv5'].W.data
        dst.conv5_red.W.data = src['conv5_red'].W.data

        # now the layers that use the picture of the face
        dst.conv1_face.W.data = src['conv1_face'].W.data
        dst.conv2_face.W.data = src['conv2_face'].W.data
        dst.conv3_face.W.data = src['conv3_face'].W.data
        dst.conv4_face.W.data = src['conv4_face'].W.data
        dst.conv5_face.W.data = src['conv5_face'].W.data
        dst.fc6_face.W.data = src['fc6_face'].W.data

        # other layers
        dst.fc7_face.W.data =src['fc7_face'].W.data
        dst.fc8_face.W.data = src['fc8_face'].W.data
        dst.importance_no_sigmoid.W.data =src['conv1'].W.data
        dst.importance_map.W.data = src['importance_no_sigmoid'].W.data
        dst.fc_0_0.W.data =src['fc_0_0'].W.data
        dst.fc_1_0.W.data = src['fc_1_0'].W.data
        dst.fc_m1_0.W.data =src['fc_m1_0'].W.data
        dst.fc_0_1.W.data =src['fc_0_1'].W.data
        dst.fc_0_m1.W.data = src['fc_0_m1'].W.data

        # the network that uses data as input
        dst.conv1.b.data = src['conv1'].b.data
        dst.conv2.b.data = src['conv2'].b.data
        dst.conv3.b.data = src['conv3'].b.data
        dst.conv4.b.data = src['conv4'].b.data
        dst.conv5.b.data = src['conv5'].b.data
        dst.conv5_red.b.data = src['conv5_red'].b.data

        # now the layers that use the picture of the face
        dst.conv1_face.b.data = src['conv1_face'].b.data
        dst.conv2_face.b.data = src['conv2_face'].b.data
        dst.conv3_face.b.data = src['conv3_face'].b.data
        dst.conv4_face.b.data = src['conv4_face'].b.data
        dst.conv5_face.b.data = src['conv5_face'].b.data
        dst.fc6_face.b.data = src['fc6_face'].b.data

        # other layers
        dst.fc7_face.b.data =src['fc7_face'].b.data
        dst.fc8_face.b.data = src['fc8_face'].b.data
        dst.importance_no_sigmoid.b.data =src['conv1'].b.data
        dst.importance_map.b.data = src['importance_no_sigmoid'].b.data
        dst.fc_0_0.b.data =src['fc_0_0'].b.data
        dst.fc_1_0.b.data = src['fc_1_0'].b.data
        dst.fc_m1_0.b.data =src['fc_m1_0'].b.data
        dst.fc_0_1.b.data =src['fc_0_1'].b.data
        dst.fc_0_m1.b.data = src['fc_0_m1'].b.data


if __name__ == "__main__":
    GF = GazeFollow()
    GF._make_npz()
