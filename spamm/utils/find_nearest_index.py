#! /usr/bin/env python

import numpy as np

def find_nearest(input_list, value):
    '''
    Find nearest entry in an array to a specified value.
    Ref: http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    Args:
        input_list (list): List of floats.
        value (float): Desired value to find closest match to in the array.
    
    Returns:
        float (float): Value closest to input value from input_list.
    '''
    
    idx = (np.abs(np.asarray(input_list, dtype = float)-value)).argmin()
    return input_list[idx]

def find_nearest_index(input_list, value):
    '''
    Find nearest entry in an array to a specified value.
    Ref: http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    Args:
        input_list (list): List of floats.
        value (float): Desired value to find closest match to in the array.
    
    Returns:
        float (float): Index of value closest to input value from input_list.
    '''
    
    idx = (np.abs(np.asarray(input_list, dtype = float) - value)).argmin()
    return idx
