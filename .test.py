#!/usr/bin/env python

'''
import emc
import numpy as np
import time
import sys

btime = time.time()
recon = emc.EMC('config.ini')
etime = time.time()
print('Setup took %f seconds\n' % (etime-btime))
for i in range(20):
    #recon.num_rot = int(360 * (1 - np.exp(-(i+1)/20)))
    #stime = time.time()
    recon.run_iteration(i)
    #etime = time.time()
    #sys.stderr.write('\rIteration %d took %f seconds' % (i+1, etime-stime))
etime = time.time()
sys.stderr.write('\n%d iterations took %f seconds\n' % (i+1, etime-btime))
'''

import numpy as np
import h5py
import cupy as cp
from cupy.cuda import runtime
from cupy.cuda import texture

import kernels

with h5py.File('data/photons.h5', 'r') as f:
    sol = f['solution'][:]

desc = texture.ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
cuarr = texture.CUDAarray(desc, 201, 201)

resdesc = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuarr)
texdesc = texture.TextureDescriptor((runtime.cudaAddressModeBorder, runtime.cudaAddressModeBorder),
                                    runtime.cudaFilterModeLinear,
                                    runtime.cudaReadModeElementType,
                                    normalizedCoords=1)
texture = texture.TextureObject(resdesc, texdesc)

view = cp.array((201, 201), dtype='f4')
bg = cp.zeros((201, 201), dtype='f4')
bsize = int(np.ceil(201/32))
angle = np.pi / 6.
scale = 1.
size = 201
log_flag = 0
kernels.slice_gen_tex((bsize, bsize), (32, 32), (texture, angle, scale, size, bg, log_flag, view))

np.save('data/rot_view.npy', view.get())

