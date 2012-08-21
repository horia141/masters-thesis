#ifndef _CODING_METHODS_H
#define _CODING_METHODS_H

#include "base_defines.h"

typedef void (*coding_method_t)(double* restrict,size_t* restrict,size_t,size_t,const double* restrict,const double* restrict,const double* restrict,size_t,const void* restrict,const double* restrict,void* restrict);

extern void  correlation(double* restrict o_coeffs,size_t* restrict o_coeffs_idx,size_t geometry,size_t word_count,const double* restrict dict,const double* restrict dict_transp,const double* restrict dict_x_dict_transp,size_t coeff_count,const void* restrict coding_params,const double* restrict observation,void* restrict coding_tmps);
extern void  matching_pursuit(double* restrict o_coeffs,size_t* restrict o_coeffs_idx,size_t geometry,size_t word_count,const double* restrict dict,const double* restrict dict_transp,const double* restrict dict_x_dict_transp,size_t coeff_count,const void* restrict coding_params,const double* restrict observation,void* restrict coding_tmps);

#endif
