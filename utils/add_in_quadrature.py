#! /usr/bin/env python

import numpy as np

def add_in_quadrature(*args):
    sqsum = 0.
    for arg in args:
        sqsum += arg**2
    err = np.sqrt(sqsum)

    return err

