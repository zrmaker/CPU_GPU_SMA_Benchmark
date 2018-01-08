#!/usr/bin/env python

import numpy as np
import time
from numba import guvectorize

@guvectorize(['void(float64[:], intp[:], float64[:])'], '(n),()->(n)')
def moving_mean_gpu(a, window_arr, out):
    window_length = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_length):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_length, len(a)):
        asum += a[i] - a[i - window_length]
        out[i] = asum / count

def moving_mean_cpu(a, window_length):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(a, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=window_length:
            moving_ave = (cumsum[i] - cumsum[i-window_length])/window_length
            moving_aves.append(moving_ave)
        else:
            moving_ave = cumsum[i]/i
            moving_aves.append(moving_ave)
    return moving_aves

window_length = 100
arr = np.arange(1e7,dtype=np.float)
st=time.time()
sma_gpu=moving_mean_gpu(arr, window_length)
# print(len(sma_gpu))
print(time.time()-st)

st=time.time()
sma_cpu=moving_mean_cpu(arr, window_length)
# print(len(sma_cpu))
print(time.time()-st)
