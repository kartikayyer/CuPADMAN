import cupy as cp

slice_gen = cp.RawKernel(r'''
    extern "C" __device__
    void gen_rot(const double quaternion[4], double rot[3][3]) {
        double q0, q1, q2, q3, q01, q02, q03, q11, q12, q13, q22, q23, q33 ;
        
        q0 = quaternion[0] ;
        q1 = quaternion[1] ;
        q2 = quaternion[2] ;
        q3 = quaternion[3] ;
        
        q01 = q0*q1 ;
        q02 = q0*q2 ;
        q03 = q0*q3 ;
        q11 = q1*q1 ;
        q12 = q1*q2 ;
        q13 = q1*q3 ;
        q22 = q2*q2 ;
        q23 = q2*q3 ;
        q33 = q3*q3 ;
        
        rot[0][0] = (1. - 2.*(q22 + q33)) ;
        rot[0][1] = 2.*(q12 + q03) ;
        rot[0][2] = 2.*(q13 - q02) ;
        rot[1][0] = 2.*(q12 - q03) ;
        rot[1][1] = (1. - 2.*(q11 + q33)) ;
        rot[1][2] = 2.*(q01 + q23) ;
        rot[2][0] = 2.*(q02 + q13) ;
        rot[2][1] = 2.*(q23 - q01) ;
        rot[2][2] = (1. - 2.*(q11 + q22)) ;
    }

    extern "C" __global__
    void slice_gen(const double *model,
                   const double *quat,
                   const double *qvals,
                   const double scale,
                   const long long num_pix,
                   const long long size,
                   const long long log_flag,
                   double *view) {
        int t = blockIdx.x * blockDim.x + threadIdx.x ;
        int i, j, cen = size/2 ;
        double rot_pix[3], rot[3][3] ;
        if (t >= num_pix)
            return ;

        gen_rot(quat, rot) ;
        
        for (i = 0 ; i < 3 ; ++i) {
            rot_pix[i] = 0. ;
            for (j = 0 ; j < 3 ; ++j) 
                rot_pix[i] += rot[i][j] * qvals[t*3 + j] ;
            rot_pix[i] += cen ;
        }

        int ix = __double2int_rd(rot_pix[0]) ;
        int iy = __double2int_rd(rot_pix[1]) ;
        int iz = __double2int_rd(rot_pix[2]) ;
        if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2) {
            if (log_flag)
                view[t] = -1.0e20 ;
            else
                view[t] = 0. ;
        
            return ;
        }

        double fx = rot_pix[0] - ix, fy = rot_pix[1] - iy, fz = rot_pix[2] - iz ;
        double cx = 1. - fx, cy = 1. - fy, cz = 1. - fz ;

        view[t] = cx*cy*cz*model[ix*size*size + iy*size + iz] + 
                  cx*cy*fz*model[ix*size*size + iy*size + iz+1] +
                  cx*fy*cz*model[ix*size*size + (iy+1)*size + iz] + 
                  cx*fy*fz*model[ix*size*size + (iy+1)*size + iz+1] +
                  fx*cy*cz*model[(ix+1)*size*size + iy*size + iz] + 
                  fx*cy*fz*model[(ix+1)*size*size + iy*size + iz+1] +
                  fx*fy*cz*model[(ix+1)*size*size + (iy+1)*size + iz] + 
                  fx*fy*fz*model[(ix+1)*size*size + (iy+1)*size + iz+1] ;
        view[t] *= scale ;
        if (log_flag) {
            if (view[t] < 1.e-20)
                view[t] = -1.0e20 ;
            else
                view[t] = log(view[t]) ;
        }
    }
    ''', 'slice_gen')

slice_merge = cp.RawKernel(r'''
    extern "C" __device__
    void gen_rot(const double quaternion[4], double rot[3][3]) {
        double q0, q1, q2, q3, q01, q02, q03, q11, q12, q13, q22, q23, q33 ;
        
        q0 = quaternion[0] ;
        q1 = quaternion[1] ;
        q2 = quaternion[2] ;
        q3 = quaternion[3] ;
        
        q01 = q0*q1 ;
        q02 = q0*q2 ;
        q03 = q0*q3 ;
        q11 = q1*q1 ;
        q12 = q1*q2 ;
        q13 = q1*q3 ;
        q22 = q2*q2 ;
        q23 = q2*q3 ;
        q33 = q3*q3 ;
        
        rot[0][0] = (1. - 2.*(q22 + q33)) ;
        rot[0][1] = 2.*(q12 + q03) ;
        rot[0][2] = 2.*(q13 - q02) ;
        rot[1][0] = 2.*(q12 - q03) ;
        rot[1][1] = (1. - 2.*(q11 + q33)) ;
        rot[1][2] = 2.*(q01 + q23) ;
        rot[2][0] = 2.*(q02 + q13) ;
        rot[2][1] = 2.*(q23 - q01) ;
        rot[2][2] = (1. - 2.*(q11 + q22)) ;
    }

    extern "C" __global__
    void slice_merge(const double *view,
                     const double *quat,
                     const double *qvals,
                     const long long num_pix,
                     const long long size,
                     double *model,
                     double *mweights) {
        int t = blockIdx.x * blockDim.x + threadIdx.x ;
        if (t >= num_pix)
            return ;

        int i, j, cen = size/2 ;
        double rot_pix[3], rot[3][3] ;
        if (t >= num_pix)
            return ;

        gen_rot(quat, rot) ;
        
        for (i = 0 ; i < 3 ; ++i) {
            rot_pix[i] = 0. ;
            for (j = 0 ; j < 3 ; ++j) 
                rot_pix[i] += rot[i][j] * qvals[t*3 + j] ;
            rot_pix[i] += cen ;
        }

        int ix = __double2int_rd(rot_pix[0]) ;
        int iy = __double2int_rd(rot_pix[1]) ;
        int iz = __double2int_rd(rot_pix[2]) ;
        if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
            return ;

        double fx = rot_pix[0] - ix, fy = rot_pix[1] - iy, fz = rot_pix[2] - iz ;
        double cx = 1. - fx, cy = 1. - fy, cz = 1. - fz ;

        atomicAdd(&model[ix*size*size + iy*size + iz], view[t]*cx*cy*cz) ;
        atomicAdd(&model[ix*size*size + iy*size + iz+1], view[t]*cx*cy*fz) ;
        atomicAdd(&model[ix*size*size + (iy+1)*size + iz], view[t]*cx*fy*cz) ;
        atomicAdd(&model[ix*size*size + (iy+1)*size + iz+1], view[t]*cx*fy*fz) ;
        atomicAdd(&model[(ix+1)*size*size + iy*size + iz], view[t]*fx*cy*cz) ;
        atomicAdd(&model[(ix+1)*size*size + iy*size + iz+1], view[t]*fx*cy*fz) ;
        atomicAdd(&model[(ix+1)*size*size + (iy+1)*size + iz], view[t]*fx*fy*cz) ;
        atomicAdd(&model[(ix+1)*size*size + (iy+1)*size + iz+1], view[t]*fx*fy*fz) ;

        atomicAdd(&mweights[ix*size*size + iy*size + iz], cx*cy*cz) ;
        atomicAdd(&mweights[ix*size*size + iy*size + iz+1], cx*cy*fz) ;
        atomicAdd(&mweights[ix*size*size + (iy+1)*size + iz], cx*fy*cz) ;
        atomicAdd(&mweights[ix*size*size + (iy+1)*size + iz+1], cx*fy*fz) ;
        atomicAdd(&mweights[(ix+1)*size*size + iy*size + iz], fx*cy*cz) ;
        atomicAdd(&mweights[(ix+1)*size*size + iy*size + iz+1], fx*cy*fz) ;
        atomicAdd(&mweights[(ix+1)*size*size + (iy+1)*size + iz], fx*fy*cz) ;
        atomicAdd(&mweights[(ix+1)*size*size + (iy+1)*size + iz+1], fx*fy*fz) ;
    }
    ''', 'slice_merge')

calc_prob_all = cp.RawKernel(r'''
    extern "C" __global__
    void calc_prob_all(const double *lview,
                       const long long ndata,
                       const int *ones,
                       const int *multi,
                       const long long *o_acc,
                       const long long *m_acc,
                       const int *p_o,
                       const int *p_m,
                       const int *c_m,
                       const double init,
                       const double *scales,
                       double *prob_r) {
        long long d, t ;
        d = blockDim.x * blockIdx.x + threadIdx.x ;
        if (d >= ndata)
            return ;

        prob_r[d] = init * scales[d] ;
        for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t)
            prob_r[d] += lview[p_o[t]] ;
        for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t)
            prob_r[d] += lview[p_m[t]] * c_m[t] ;
    }
    ''', 'calc_prob_all')

merge_all = cp.RawKernel(r'''
    extern "C" __global__
    void merge_all(const double *prob_r,
                   const long long ndata,
                   const int *ones,
                   const int *multi,
                   const long long *o_acc,
                   const long long *m_acc,
                   const int *p_o,
                   const int *p_m,
                   const int *c_m,
                   double *view) {
        long long d, t ;
        d = blockDim.x * blockIdx.x + threadIdx.x ;
        if (d >= ndata)
            return ;

        for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t)
            atomicAdd(&view[p_o[t]], prob_r[d]) ;
        for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t)
            atomicAdd(&view[p_m[t]], prob_r[d] * c_m[t]) ;
    }
    ''', 'merge_all')

