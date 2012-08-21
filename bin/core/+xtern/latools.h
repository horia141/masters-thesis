#ifndef _LATOOLS_H
#define _LATOOLS_H

#include "base_defines.h"

extern void  dict_obs_inner_product(double* restrict o_similarities,size_t geometry,size_t word_count,const double* restrict dict,const double* restrict dict_transp,const double* restrict observation);
extern void  fill_idx_1n(size_t* restrict o_idx,size_t count);
extern void  partition_greatest(double* restrict o_coeffs,size_t* restrict o_coeffs_idx,size_t count,size_t greatest_count);
extern void  sort_by_abs_coeffs(double* restrict o_coeffs,size_t* restrict o_coeffs_idx,size_t count);
extern void  sort_by_idxs(double* restrict o_coeffs,size_t* restrict o_coeffs_idx,size_t count);

#endif
