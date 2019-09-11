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

import time
import numpy as np
import h5py
import cupy as cp
from cupy.cuda import runtime
from cupy.cuda import texture

import kernels

with h5py.File('data/photons.h5', 'r') as f:
    sol = f['solution'][:]

# Testing slice_gen_tex
desc = texture.ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
cuarr = texture.CUDAarray(desc, 201, 201)
cuarr.copy_from(sol.astype('f4'))

resdesc = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuarr)
texdesc = texture.TextureDescriptor((runtime.cudaAddressModeBorder, runtime.cudaAddressModeBorder),
                                    runtime.cudaFilterModeLinear,
                                    runtime.cudaReadModeElementType,
                                    normalizedCoords=True)
texture = texture.TextureObject(resdesc, texdesc)

view = cp.empty((201, 201), dtype='f4')
bg = cp.zeros((201, 201), dtype='f4')
bsize = int(np.ceil(201/32))
scale = 1.
size = 201
log_flag = 0
stime = time.time()
for i in range(10000):
    angle = i * np.pi / 180.
    kernels.slice_gen_tex((bsize, bsize), (32, 32), (texture, angle, scale, size, bg, log_flag, view))
etime = time.time()
print('Time taken with texture: %f s'%(etime-stime))

#np.save('data/rot_view.npy', view.get())

# Testing normal slice_gen
view = cp.empty((201, 201), dtype='f4')
bg = cp.zeros((201, 201), dtype='f4')
model = cp.array(sol.astype('f8'))

stime = time.time()
for i in range(10000):
    angle = i * np.pi / 180.
    kernels.slice_gen((bsize, bsize), (32, 32), (model, angle, scale, size, bg, log_flag, view))
etime = time.time()
print('Time taken without texture: %f s'%(etime-stime))
