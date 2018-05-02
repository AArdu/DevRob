#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:39:31 2018

@author: gevelingbm
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import euclidean
from scipy.linalg import pinv


"""
Start with 400 (?) random movements and detect the end location of the movement.
We have 2 matrices than. One 4x400 with the joint positions and one 2x400 with the end coordinates. 
"""


# initialize
DOF = 4
nrSimulations = 400 
nrHidden = 150
nrDimensions = 2
stepsize = 5 # article uses 10mm, we only have pixels, so we guessed 5 would be fine

testInput = np.random.rand(nrSimulations, DOF) 
testOutput = np.random.rand(nrSimulations, nrDimensions)

def TrainRBF(nrSimulations, nrHidden, Input, Output): 
    
    # Choose the centers of the clusters at random
    # The output should be a DOF x nrHidden matrix
    # TODO: This should be done by OLS, but is random now. 
    mu = np.random.permutation(Input)[0:nrHidden]
    
    # Determine cluster widths
    # TODO: this should be done by OLS as well, but for now, we use the same width for each cluster
    dmax = np.amax(pdist(mu))
    spread = dmax/np.sqrt(2*nrHidden)   
    
    # determine the weights. The gaussian basis function is used (the normal one, not the one in the article, because I am not sure if this is the same as the normal one). 
    phi = np.zeros((nrHidden,nrSimulations))
    for i in range(nrHidden):
        for j in range (nrSimulations):
            phi[i,j] = np.exp(-(pdist(np.array([Input.transpose()[:,j],mu.transpose()[:,i]]))**2)/(2*(spread**2)))    
    w = np.dot(pinv(phi).transpose(), Output)
    
    return (mu, spread, w)


def useRBF(jointVector, w, mu, spread, nrHidden):
    #Calculate the activations of the hidden nodes
    hidden = np.zeros([1,nrHidden])
    for k in range(nrHidden):
        hidden[0,k] = np.exp(-(euclidean(jointVector, mu[k,:])**2)/(2*(spread**2)));
              
    # Calculate the output
    output = np.dot(hidden,w)

    return output

"""
Instead of using TrainRBF, we can train it in Matlab, with newrb.
We can use the RBF with the useRBF function if we get w and mu from matlab.
We can also run in in matlab with sim(network, input). Should be something like: 
import matlab.engine
eng = matlab.engine.start_matlab()
output = eng.sim(network, input, nargout = nrDimensions)
"""


def reach(target, stepsize, mu, spead, w, nrHidden):
    # TODO: get initial position of joints of NAO.
    jointVector= (0,0,0,0,0,0 )# this should be the initial position of the joints
    location = useRBF(jointVector, w, mu, spread, nrHidden)     # current location of the hand
    counter  = 0    # counts how many movement steps are made
    alfa = 0  
    
    # move till correct location is reached or 25 movements are made 
    while(alfa != 1 & counter <= 25):
        counter += 1
        if(euclidean(target, location)> stepsize):
            alfa = stepsize/euclidean(target, location)
        else: 
            alfa = 1
     
        jacobian = calculateJacobian(DOF, nrHidden, jointVector, mu, spread)
        invJacobian = pinv(jacobian)
        difLocation = alfa*(target-location)
        jointVector = jointVector + invJacobian*difLocation
        # TODO: move the NAO
        location = useRBF(jointVector, w, mu, spread, nrHidden)  
        
        
def calculateJacobian(DOF, nrHidden, jointVector, mu, spread):
    #Calculate the activations of the hidden nodes
    hidden = np.zeros([DOF ,nrHidden])
    for t in range(DOF): 
        for k in range(nrHidden):
            # hidden layer now uses the derivative of the Gaussian
            hidden[t,k] = -(((jointVector[t]-mu[k,t])*np.exp((-(jointVector[1]-mu[k,1])**2-(jointVector[2]-mu[k,2])**2-(jointVector[3]-mu[k,3])**2)-(jointVector[0]-mu[k,0])**2/(2*(spread**2))))/(spread**2)) 
             
    # Calculate the output
    output = np.dot(hidden,w)

    return output
   

[mu, spread, w] = TrainRBF(nrSimulations, nrHidden, testInput, testOutput)
outputRBF = useRBF(np.random.rand(4), w, mu, spread, nrHidden)
