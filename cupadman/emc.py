#!/usr/bin/env python

'''EMC reconstructor object and script'''

import sys
import os
import argparse
import configparser
import time

import numpy as np
import h5py
from mpi4py import MPI
import cupy as cp

from cupadman import Detector, CDataset, Quaternion
P_MIN = 1.e-6
MEM_THRESH = 0.8

class EMC():
    '''Reconstructor object using parameters from config file

    Args:
        config_file (str): Path to configuration file

    The appropriate CUDA device must be selected before initializing the class.
    Can be used with mpirun, in which case work will be divided among ranks.
    '''
    def __init__(self, config_file, num_streams=4):
        '''Parse config file and setup reconstruction

        One can just use run_iteration() after this
        '''
        # Get system properties
        self.num_streams = num_streams
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.num_proc = self.comm.size
        self.mem_size = cp.cuda.Device(cp.cuda.runtime.getDevice()).mem_info[1]

        # Parse config file
        config = configparser.ConfigParser()
        config.read(config_file)

        self.num_div = config.getint('emc', 'num_div')
        self.num_modes = config.getint('emc', 'num_modes', fallback=1)
        self.detector_file = os.path.join(os.path.dirname(config_file),
                                         config.get('emc', 'in_detector_file'))
        self.photons_file = os.path.join(os.path.dirname(config_file),
                                         config.get('emc', 'in_photons_file'))
        self.output_folder = os.path.join(os.path.dirname(config_file),
                                          config.get('emc', 'output_folder', fallback='data/'))
        self.log_file = os.path.join(os.path.dirname(config_file),
                                     config.get('emc', 'log_file', fallback='EMC.log'))
        model_fname = config.get('emc', 'start_model_file', fallback=None)
        self.need_scaling = config.getboolean('emc', 'need_scaling', fallback=False)

        # Setup reconstruction
        stime = time.time()

        # Note the following three structs have data in CPU memory
        self.det = Detector(self.detector_file)
        self.dset = CDataset(self.photons_file, self.det)
        self.quat = Quaternion(self.num_div)

        self.quat.divide(self.rank, self.num_proc, self.num_modes)
        self.size = int(2*np.ceil(np.linalg.norm(self.det.qvals, axis=1).max()) + 3)
        self._move_to_gpu()

        etime = time.time()
        sys.stdout.flush()
        sys.stderr.flush()

        # Get CUDA kernels
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(script_dir+'/kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen = kernels.get_function('slice_gen')
        self.k_slice_merge = kernels.get_function('slice_merge')
        self.k_calc_prob_all = kernels.get_function('calc_prob_all')
        self.k_merge_all = kernels.get_function('merge_all')

        if self.rank == 0:
            print('%d frames with %.3f photons/frame (%.3f s) (%.2f MB)' % \
                    (self.dset.num_data, self.dset.mean_count, etime-stime, self.dset_mem/1024**2))
            sys.stdout.flush()
        self.model = np.empty(3*(self.size,))
        if self.rank == 0:
            if model_fname is None:
                print('Random start')
                self.model[:] = np.random.random((self.size,)*3) * self.dset.mean_count / self.dset.num_pix
            else:
                print('Parsing model from', model_fname)
                with h5py.File(model_fname, 'r') as fptr:
                    self.model[:] = fptr['intens'][0]
                #self.model[:] = np.load(model_fname)
        self.comm.Bcast([self.model, MPI.DOUBLE], root=0)
        self.mweights = np.zeros(3*(self.size,))
        if self.need_scaling:
            self.dset.calc_frame_counts()
            self.scales = cp.array(self.dset.fcounts) / self.dset.mean_count
            self.beta_start = cp.array(np.exp(-6.5 * pow(self.dset.fcounts * 1.e-5, 0.15))) # Empirical
        else:
            self.scales = cp.ones(self.dset.num_data, dtype='f8')
            self.beta_start = cp.array(np.exp(-6.5 * pow(np.ones(self.dset.num_data)*self.dset.mean_count * 1.e-5, 0.15))) # Empirical
        print('Starting beta =', self.beta_start.mean())
        self.prob = cp.array([])

        self.bsize_pixel = int(np.ceil(self.det.num_pix/32.))
        self.bsize_data = int(np.ceil(self.dset.num_data/32.))
        self.stream_list = [cp.cuda.Stream() for _ in range(self.num_streams)]

    def run_iteration(self, iternum=1):
        '''Run one iterations of EMC algorithm

        Args:
            iternum (int): Iteration number

        Current guess is assumed to be in self.model, which is updated. If scaling is included,
        the scale factors are in self.scales.
        '''

        stime = time.time()
        mem_frac = self.quat.num_rot_p*self.dset.num_data*8/ (self.mem_size - self.dset_mem)
        num_blocks = int(np.ceil(mem_frac / MEM_THRESH))
        block_sizes = np.array([self.dset.num_data // num_blocks] * num_blocks)
        block_sizes[0:self.dset.num_data % num_blocks] += 1
        if len(block_sizes) > 1 and self.rank == 0: print(block_sizes, 'frames in each block')

        if self.prob.shape != (self.quat.num_rot_p, block_sizes.max()):
            self.prob = cp.empty((self.quat.num_rot_p, block_sizes.max()), dtype='f8')
        views = cp.empty((self.num_streams, self.det.num_pix), dtype='f8')
        dmodel = cp.array(self.model)
        dmweights = cp.array(self.mweights)
        factor = 2 ** ((iternum-1) // 10)
        self.beta = self.beta_start * factor
        #mp = cp.get_default_memory_pool()
        #print('Mem usage: %.2f MB / %.2f MB' % (mp.total_bytes()/1024**2, self.mem_size/1024**2))
        self._calculate_rescale(dmodel, views)

        b_start = 0
        for b in block_sizes:
            drange = (b_start, b_start + b)
            self._calculate_prob(dmodel, views, drange)
            self._normalize_prob()
            self._update_model(views, dmodel, dmweights, drange)
            b_start += b
        diff = self._normalize_model(dmodel, dmweights, iternum)
        etime = time.time()
        if self.rank == 0:
            self._write_log(iternum, etime-stime, diff)

    def _calculate_rescale(self, dmodel, views):
        self.vsum = cp.zeros(self.quat.num_rot, dtype='f8')
        total = 0.

        for i, r in enumerate(range(self.rank, self.quat.num_rot, self.num_proc)):
            snum = i % self.num_streams
            self.stream_list[snum].use()
            self.k_slice_gen((self.bsize_pixel,), (32,),
                    (dmodel, self.quats[r], self.pixvals,
                     self.dmask, 0., self.det.num_pix,
                     self.size, views[snum]))
            self.vsum[r] = views[snum][self.det.raw_mask==0].sum()
            total += self.vsum[r] * self.quat.quats[r,4]
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()

        self.rescale = self.dset.mean_count / total
        #print('rescale =', self.rescale)

    def _calculate_prob(self, dmodel, views, drange):
        s = drange[0]
        e = drange[1]
        num_data_b = e - s
        self.bsize_data = int(np.ceil(num_data_b/32.))

        for i, r in enumerate(range(self.rank, self.quat.num_rot, self.num_proc)):
            snum = i % self.num_streams
            self.stream_list[snum].use()
            self.k_slice_gen((self.bsize_pixel,), (32,),
                    (dmodel, self.quats[r], self.pixvals,
                     self.dmask, 1., self.det.num_pix,
                     self.size, views[snum]))
            #initval = float(cp.log(self.quats[r,4])) - float(views[snum].sum())
            #initval = - float(views[snum].sum())
            #initval = 0.
            initval = float(cp.log(self.quats[r,4]) - self.vsum[r] * self.rescale)
            self.k_calc_prob_all((self.bsize_data,), (32,),
                    (views[snum], num_data_b,
                     self.ones[s:e], self.multi[s:e],
                     self.ones_accum[s:e], self.multi_accum[s:e],
                     self.place_ones, self.place_multi, self.count_multi,
                     self.dmask, initval, self.scales[s:e], self.prob[i]))
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()

    def _normalize_prob(self):
        max_exp_p = self.prob.max(0).get()
        rmax_p = (self.prob.argmax(axis=0) * self.num_proc + self.rank).astype('i4').get()
        max_exp = np.empty_like(max_exp_p)
        self.rmax = np.empty_like(rmax_p)

        self.comm.Allreduce([max_exp_p, MPI.DOUBLE], [max_exp, MPI.DOUBLE], op=MPI.MAX)
        rmax_p[max_exp_p != max_exp] = -1
        self.comm.Allreduce([rmax_p, MPI.INT], [self.rmax, MPI.INT], op=MPI.MAX)
        max_exp = cp.array(max_exp)

        self.prob = cp.exp(self.beta*cp.subtract(self.prob, max_exp, self.prob), self.prob)
        psum_p = self.prob.sum(0).get()
        psum = np.empty_like(psum_p)
        self.comm.Allreduce([psum_p, MPI.DOUBLE], [psum, MPI.DOUBLE], op=MPI.SUM)
        self.prob = cp.divide(self.prob, cp.array(psum), self.prob)

    def _update_model(self, views, dmodel, dmweights, drange):
        p_norm = self.prob.sum(1)
        h_p_norm = p_norm.get()
        s = drange[0]
        e = drange[1]
        num_data_b = e - s

        dmodel[:] = 0
        dmweights[:] = 0
        for i, r in enumerate(range(self.rank, self.quat.num_rot, self.num_proc)):
            if h_p_norm[i] == 0.:
                continue
            snum = i % self.num_streams
            self.stream_list[snum].use()
            views[snum,:] = 0
            self.k_merge_all((self.bsize_data,), (32,),
                    (self.prob[i], num_data_b,
                     self.ones[s:e], self.multi[s:e],
                     self.ones_accum[s:e], self.multi_accum[s:e],
                     self.place_ones, self.place_multi, self.count_multi,
                     self.dmask, views[snum]))
            #views[snum] = views[snum] / p_norm[i] - self.dset.bg
            views[snum] = views[snum] / p_norm[i]
            self.k_slice_merge((self.bsize_pixel,), (32,),
                    (views[snum], self.quats[r],
                     self.pixvals, self.dmask, self.det.num_pix,
                     self.size, dmodel, dmweights))
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()

    def _normalize_model(self, dmodel, dmweights, iternum):
        old_model = np.copy(self.model)
        self.model = dmodel.get()
        self.mweights = dmweights.get()
        if self.rank == 0:
            self.comm.Reduce(MPI.IN_PLACE, [self.model, MPI.DOUBLE], root=0, op=MPI.SUM)
            self.comm.Reduce(MPI.IN_PLACE, [self.mweights, MPI.DOUBLE], root=0, op=MPI.SUM)
            self.model[self.mweights > 0] /= self.mweights[self.mweights > 0]

            self._save_output(iternum)
            diff = np.linalg.norm(self.model - old_model)
        else:
            self.comm.Reduce([self.model, MPI.DOUBLE], None, root=0, op=MPI.SUM)
            self.comm.Reduce([self.mweights, MPI.DOUBLE], None, root=0, op=MPI.SUM)
            diff = 0.
        self.comm.Bcast([self.model, MPI.DOUBLE], root=0)
        return diff

    def _save_output(self, iternum):
        fptr = h5py.File(self.output_folder + '/output_%.3d.h5'%iternum, 'w')
        fptr['intens'] = self.model[np.newaxis]
        fptr['inter_weight'] = self.mweights[np.newaxis]
        fptr['orientations'] = self.rmax
        fptr['scale'] = self.scales.get()
        fptr.close()

    def _write_log(self, iternum, itertime, norm):
        if iternum == 1:
            fptr = open(self.log_file, 'w')
            fptr.write('Cryptotomography with the EMC algorithm using MPI+CUDA\n\n')
            fptr.write('Data parameters:\n\tnum_data = %d\n\tmean_count = %f\n\n' % (self.dset.num_data, self.dset.mean_count))
            fptr.write('System size:\n\tnum_rot = %d\n\tnum_pix = %d/%d\n\tvolume = %d x %d x %d x %d\n\n' % (self.quat.num_rot, (self.det.raw_mask==0).sum(), self.det.num_pix, self.num_modes, self.size, self.size ,self.size))
            fptr.write('Reconstruction parameters:\n\t')
            fptr.write('num_proc = %d\n\t'%self.num_proc)
            fptr.write('alpha = 0.0\n\t')
            fptr.write('beta = %f\n\t'%self.beta.mean())
            fptr.write('need_scaling = %s\n\n'%('yes' if self.need_scaling else 'no'))
            fptr.write('Iter  time     rms_change   info_rate  log-likelihood  num_rot  beta\n')
        else:
            fptr = open(self.log_file, 'a')

        fptr.write('%-6d%-.2e %e %f   %e    %-8d %f\n' % (iternum, itertime, norm, 0, 0, self.quat.num_rot, self.beta.mean()))
        fptr.close()

    def _move_to_gpu(self):
        '''Move detector, dataset and quaternions to GPU'''
        self.dmask = cp.array(self.det.raw_mask)
        self.pixvals = cp.array(np.concatenate((self.det.qvals, self.det.corr[:,np.newaxis]), axis=1).ravel())

        self.quats = cp.array(self.quat.quats)

        init_mem = cp.get_default_memory_pool().used_bytes()
        self.ones = cp.array(self.dset.ones)
        self.multi = cp.array(self.dset.multi)
        self.ones_accum = cp.array(self.dset.ones_accum)
        self.multi_accum = cp.array(self.dset.multi_accum)
        self.place_ones = cp.array(self.dset.place_ones)
        self.place_multi = cp.array(self.dset.place_multi)
        self.count_multi = cp.array(self.dset.count_multi)
        self.dset_mem = cp.get_default_memory_pool().used_bytes() - init_mem

def main():
    '''Parses command line arguments and launches EMC reconstruction'''
    import socket
    parser = argparse.ArgumentParser(description='Cryptotomography with the EMC algorithm using MPI+CUDA')
    parser.add_argument('num_iter', type=int,
                        help='Number of iterations')
    parser.add_argument('-c', '--config_file', default='config.ini',
                        help='Path to configuration file (default: config.ini)')
    parser.add_argument('-d', '--devices', default=None,
                        help='Comma-separated list of device numbers')
    parser.add_argument('-s', '--streams', type=int, default=4,
                        help='Number of streams to use (default=4)')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    num_proc = comm.size
    if args.devices is None:
        if num_proc == 1:
            print('Running on default device 0')
            sys.stdout.flush()
        else:
            print('Require "devices" option when using multiple processes')
            sys.exit(1)
    else:
        dev = args.devices.split(',')
        if len(dev) != num_proc:
            print('Number of devices (with repetition) must equal number of MPI ranks')
            sys.exit(1)
        cp.cuda.Device(int(dev[rank])).use()

    recon = EMC(args.config_file, num_streams=args.streams)
    if rank == 0:
        print('\nIter  time(s)  change       beta')
        sys.stdout.flush()
        avgtime = 0.
        numavg = 0
    for i in range(args.num_iter):
        m0 = cp.array(recon.model)
        stime = time.time()
        recon.run_iteration(i+1)
        etime = time.time()
        if rank == 0:
            norm = float(cp.linalg.norm(cp.array(recon.model) - m0))
            print('%-6d%-.2e %e %e' % (i+1, etime-stime, norm, recon.beta.mean()))
            sys.stdout.flush()
            if i > 0:
                avgtime += etime-stime
                numavg += 1
    if rank == 0 and numavg > 0:
        print('%.4e s/iteration on average' % (avgtime / numavg))

if __name__ == '__main__':
    main()
