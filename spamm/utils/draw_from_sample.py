#! /usr/bin/env python

import numpy as np

def gaussian(minimum, maximum, num=None):
    mu = minimum + ((maximum - minimum) / 2.)
    sigma = (maximum - mu) * .341

    samples = np.random.normal(mu, sigma, num)

    return samples
    
