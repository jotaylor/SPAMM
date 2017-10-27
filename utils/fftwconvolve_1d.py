#! /usr/bin/env python

def fftwconvolve_1d(in1, in2):
    '''
    This code taken from:
     https://stackoverflow.com/questions/32028979/speed-up-for-loop-in-convolution-for-numpy-3d-array
     and the resulting output is the full discrete linear convolution of the 
     inputs(i.e. This returns the convolution at each point of overlap), 
     which includes additional terms at the start and end of the array such 
     that if A has size N and B has size M when covolved the size is N+M-1. 
     At the end-points of the convolution, the signals do not overlap completely, 
     and boundary effects may be seen.  
    '''
    outlen = in1.shape[-1] + in2.shape[-1] - 1 
    origlen = in1.shape[-1]
    n = next_fast_len(outlen) 
    tr1 = pyfftw.interfaces.numpy_fft.rfft(in1, n) 
    tr2 = pyfftw.interfaces.numpy_fft.rfft(in2, n) 
    sh = np.broadcast(tr1, tr2).shape 
    dt = np.common_type(tr1, tr2) 
    pr = pyfftw.n_byte_align_empty(sh, 16, dt) 
    np.multiply(tr1, tr2, out=pr) 
    out = pyfftw.interfaces.numpy_fft.irfft(pr, n) 

    index_low = int(outlen/2.)-int(np.floor(origlen/2))
    index_high = int(outlen/2.)+int(np.ceil(origlen/2))
    # these two lines find the central indices of the resulting array
    return out[..., index_low:index_high].copy() #returns an array the same length as the input 1 and is the appropriate output. Boundary effects are still visible and when overlap is not complete zero values are assumed I believe.

