#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:25:41 2018

@author: gevelingbm
"""
import scipy.io 
import numpy as np

# filename = '/Users/gevelingbm/Downloads/rbfweights-2'

def point(coordinates, filename):
      
    bandw = scipy.io.loadmat(filename)
    weights1 = np.array(list(bandw.get('weigth1')))
    weights2 = np.array(list(bandw.get('weigth2')))
    biases1 = np.array(list(bandw.get('biases1')))
    biases2 = np.array(list(bandw.get('biases2')))
    
    h_input = (np.dot(coordinates, weights1.transpose())* biases1.transpose())
    h_act = np.exp(-(h_input**2))
    
    o_input = np.dot(h_act,weights2.transpose()) + biases2.transpose()
    o_act = o_input

    return o_act

# movements = point((200,300), filename)
