#! /usr/bin/env python

import numpy as np

def runningMeanFast(x, N):
    '''
    Calculate the running mean of an array given a window.
    Ref: http://stackoverflow.com/questions/13728392/moving-average-or-running-mean

    Args:
        x (array-like): Data array
        N (int): Window width

    Returns:
        An array of averages over each window.
            
    '''
    
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

