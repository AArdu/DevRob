import numpy as np
import scipy.io


def point(coordinates, filename):
    """Reads the RBF created with MATLAB and generates a prediction"""
    bandw = scipy.io.loadmat(filename)
    weights1 = np.array(list(bandw.get('weigth1')))
    weights2 = np.array(list(bandw.get('weigth2')))
    biases1 = np.array(list(bandw.get('biases1')))
    biases2 = np.array(list(bandw.get('biases2')))

    h_input = (np.dot(coordinates, weights1.transpose()) * biases1.transpose())
    h_act = np.exp(-(h_input**2))

    o_input = np.dot(h_act, weights2.transpose()) + biases2.transpose()
    o_act = o_input

    return o_act
