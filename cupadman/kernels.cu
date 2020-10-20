extern "C" {

#define PROB_MIN 1.e-6
typedef unsigned char uint8_t ;

__device__
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

__device__
void gen_rot2d(const double angle, double rot[2][2]) {
	double c = cos(angle) ;
	double s = sin(angle) ;

	rot[0][0] = c ;
	rot[0][1] = -s ;
	rot[1][0] = s ;
	rot[1][1] = c ;
}

__global__
void slice_gen3d(const double *model, const double *quat, const double *pixvals, const uint8_t *mask,
                 const double scale, const long long num_pix, const long long size, double *view) {
	int t = blockIdx.x * blockDim.x + threadIdx.x ;
	if (t >= num_pix)
		return ;
	int i, j, cen = size/2 ;
	double rot_pix[3], rot[3][3] ;

	gen_rot(quat, rot) ;

	for (i = 0 ; i < 3 ; ++i) {
		rot_pix[i] = 0. ;
		for (j = 0 ; j < 3 ; ++j)
			rot_pix[i] += rot[i][j] * pixvals[t*4 + j] ;
		rot_pix[i] += cen ;
	}

	int ix = __double2int_rd(rot_pix[0]) ;
	int iy = __double2int_rd(rot_pix[1]) ;
	int iz = __double2int_rd(rot_pix[2]) ;
	if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2 || iz < 0 || iz > size - 2) {
		if (scale != 0.)
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
	view[t] *= pixvals[t*4 + 3] ;
	if (scale != 0.) {
		if (view[t] < 1.e-20)
			view[t] = -1.0e20 ;
		else
			view[t] = log(view[t] * scale) ;
	}
}

__global__
void slice_gen2d(const double *model, const double *quat, const double *pixvals, const uint8_t *mask,
                 const double scale, const long long num_pix, const long long size, double *view) {
	int t = blockIdx.x * blockDim.x + threadIdx.x ;
	if (t >= num_pix)
		return ;
	int i, j, cen = size/2 ;
	double rot_pix[2], rot[2][2] ;

	gen_rot2d(quat[0], rot) ;

	for (i = 0 ; i < 2 ; ++i) {
		rot_pix[i] = 0. ;
		for (j = 0 ; j < 2 ; ++j)
			rot_pix[i] += rot[i][j] * pixvals[t*4 + j] ;
		rot_pix[i] += cen ;
	}

	int ix = __double2int_rd(rot_pix[0]) ;
	int iy = __double2int_rd(rot_pix[1]) ;
	if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2) {
		if (scale != 0.)
			view[t] = -1.0e20 ;
		else
			view[t] = 0. ;

		return ;
	}
	double fx = rot_pix[0] - ix, fy = rot_pix[1] - iy ;
	double cx = 1. - fx, cy = 1. - fy ;

	view[t] = cx*cy*model[ix*size + iy] +
			  cx*fy*model[ix*size + (iy+1)] +
			  fx*cy*model[(ix+1)*size + iy] +
			  fx*fy*model[(ix+1)*size + (iy+1)] ;
	view[t] *= pixvals[t*4 + 3] ;
	if (scale != 0.) {
		if (view[t] < 1.e-20)
			view[t] = -1.0e20 ;
		else
			view[t] = log(view[t] * scale) ;
	}
}

__global__
void slice_merge3d(const double *view, const double *quat, const double *pixvals, const uint8_t *mask,
                 const long long num_pix, const long long size, double *model, double *mweights) {
	int t = blockIdx.x * blockDim.x + threadIdx.x ;
	if (t >= num_pix)
		return ;

	// Skip over bad pixels
	if (mask[t] > 1)
		return ;

	int i, j, cen = size/2 ;
	double rot_pix[3], rot[3][3] ;
	if (t >= num_pix)
		return ;

	gen_rot(quat, rot) ;

	for (i = 0 ; i < 3 ; ++i) {
		rot_pix[i] = 0. ;
		for (j = 0 ; j < 3 ; ++j)
			rot_pix[i] += rot[i][j] * pixvals[t*4 + j] ;
		rot_pix[i] += cen ;
	}

	int ix = __double2int_rd(rot_pix[0]) ;
	int iy = __double2int_rd(rot_pix[1]) ;
	int iz = __double2int_rd(rot_pix[2]) ;
	if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2 || iz < 0 || iz > size - 2)
		return ;

	double fx = rot_pix[0] - ix, fy = rot_pix[1] - iy, fz = rot_pix[2] - iz ;
	double cx = 1. - fx, cy = 1. - fy, cz = 1. - fz ;
	double val = view[t] / pixvals[t*4 + 3] ;

	atomicAdd(&model[ix*size*size + iy*size + iz], val*cx*cy*cz) ;
	atomicAdd(&model[ix*size*size + iy*size + iz+1], val*cx*cy*fz) ;
	atomicAdd(&model[ix*size*size + (iy+1)*size + iz], val*cx*fy*cz) ;
	atomicAdd(&model[ix*size*size + (iy+1)*size + iz+1], val*cx*fy*fz) ;
	atomicAdd(&model[(ix+1)*size*size + iy*size + iz], val*fx*cy*cz) ;
	atomicAdd(&model[(ix+1)*size*size + iy*size + iz+1], val*fx*cy*fz) ;
	atomicAdd(&model[(ix+1)*size*size + (iy+1)*size + iz], val*fx*fy*cz) ;
	atomicAdd(&model[(ix+1)*size*size + (iy+1)*size + iz+1], val*fx*fy*fz) ;

	atomicAdd(&mweights[ix*size*size + iy*size + iz], cx*cy*cz) ;
	atomicAdd(&mweights[ix*size*size + iy*size + iz+1], cx*cy*fz) ;
	atomicAdd(&mweights[ix*size*size + (iy+1)*size + iz], cx*fy*cz) ;
	atomicAdd(&mweights[ix*size*size + (iy+1)*size + iz+1], cx*fy*fz) ;
	atomicAdd(&mweights[(ix+1)*size*size + iy*size + iz], fx*cy*cz) ;
	atomicAdd(&mweights[(ix+1)*size*size + iy*size + iz+1], fx*cy*fz) ;
	atomicAdd(&mweights[(ix+1)*size*size + (iy+1)*size + iz], fx*fy*cz) ;
	atomicAdd(&mweights[(ix+1)*size*size + (iy+1)*size + iz+1], fx*fy*fz) ;
}

__global__
void slice_merge2d(const double *view, const double *quat, const double *pixvals, const uint8_t *mask,
                 const long long num_pix, const long long size, double *model, double *mweights) {
	int t = blockIdx.x * blockDim.x + threadIdx.x ;
	if (t >= num_pix)
		return ;

	// Skip over bad pixels
	if (mask[t] > 1)
		return ;

	int i, j, cen = size/2 ;
	double rot_pix[2], rot[2][2] ;
	if (t >= num_pix)
		return ;

	gen_rot2d(quat[0], rot) ;

	for (i = 0 ; i < 2 ; ++i) {
		rot_pix[i] = 0. ;
		for (j = 0 ; j < 2 ; ++j)
			rot_pix[i] += rot[i][j] * pixvals[t*4 + j] ;
		rot_pix[i] += cen ;
	}

	int ix = __double2int_rd(rot_pix[0]) ;
	int iy = __double2int_rd(rot_pix[1]) ;
	if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
		return ;

	double fx = rot_pix[0] - ix, fy = rot_pix[1] - iy ;
	double cx = 1. - fx, cy = 1. - fy ;
	double val = view[t] / pixvals[t*4 + 3] ;

	atomicAdd(&model[ix*size + iy], val*cx*cy) ;
	atomicAdd(&model[ix*size + (iy+1)], val*cx*fy) ;
	atomicAdd(&model[(ix+1)*size + iy], val*fx*cy) ;
	atomicAdd(&model[(ix+1)*size + (iy+1)], val*fx*fy) ;

	atomicAdd(&mweights[ix*size + iy], cx*cy) ;
	atomicAdd(&mweights[ix*size + (iy+1)], cx*fy) ;
	atomicAdd(&mweights[(ix+1)*size + iy], fx*cy) ;
	atomicAdd(&mweights[(ix+1)*size + (iy+1)], fx*fy) ;
}

__global__
void calc_prob_all(const double *lview, const long long ndata, const uint8_t *blacklist, const int *ones, const int *multi,
                   const long long *o_acc, const long long *m_acc, const int *p_o, const int *p_m,
                   const int *c_m, const uint8_t *mask, const double *initvals, double *prob_r) {
	long long d, t ;
	int pixel ;
	d = blockDim.x * blockIdx.x + threadIdx.x ;
	if (d >= ndata)
		return ;
	if (blacklist[d] == 1)
		return ;

	prob_r[d] = initvals[d] ;
	for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t) {
		pixel = p_o[t] ;
		if (mask[pixel] < 1)
			prob_r[d] += lview[pixel] ;
	}
	for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t) {
		pixel = p_m[t] ;
		if (mask[pixel] < 1)
			prob_r[d] += lview[pixel] * c_m[t] ;
	}
}

__global__
void merge_all(const double *prob_r, const long long ndata, const uint8_t *blacklist, const int *ones, const int *multi,
               const long long *o_acc, const long long *m_acc, const int *p_o, const int *p_m,
               const int *c_m, const uint8_t *mask, double *view) {
	long long d, t ;
	int pixel ;
	d = blockDim.x * blockIdx.x + threadIdx.x ;
	if (d >= ndata)
		return ;
	if (blacklist[d] == 1)
		return ;
	if (prob_r[d] < 1.e-6)
		return ;

	for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t){
		pixel = p_o[t] ;
		if (mask[pixel] < 2)
			atomicAdd(&view[pixel], prob_r[d]) ;
	}
	for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t) {
		pixel = p_m[t] ;
		if (mask[pixel] < 2)
			atomicAdd(&view[pixel], prob_r[d] * c_m[t]) ;
	}
}

__global__
void norm_prob(const double *beta, const double *max_exp, const double *psum, const double *qweight,
               const double *vsum, const uint8_t *blacklist, const long long nrot, const long long ndata, 
               double* __restrict__ prob, double *s_norm, double *likelihood, double *info) {
	long long d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;
	if (blacklist[d] == 1)
		return ;
    long long r ;
    double temp, pval ;

    s_norm[d] = 0. ;
    likelihood[d] = 0. ;
    info[d] = 0. ;
    for (r = 0 ; r < nrot ; ++r) {
        // Uncorrected log-probability
        temp = prob[r*ndata + d] ;

        // Normalized probability after exponentiation
        pval = exp(beta[d] * (temp - max_exp[d])) / psum[d] ;
        prob[r*ndata + d] = pval ;

        // Calculate likelihood
        //likelihood[d] += pval * (temp - sum_fact[d]) ;
        likelihood[d] += pval * temp ;

        // Calculate s_norm (denominator for scale factor update) [without rescale]
        s_norm[d] += pval * vsum[r] ;

        // Skip if probability is very low (saves time)
        if (pval < PROB_MIN)
            continue ;

        // Calculate mutual info
        info[d] += pval * log(pval / qweight[r]) ;
    }
}

} // extern C
