#ifndef _LATOOLS_H
#define _LATOOLS_H

#include "base_defines.h"

extern void  fill_idx_1n(size_t* restrict o_idx,size_t count);
extern void  sort_by_abs_coeffs(double* restrict o_coeffs,size_t* restrict o_coeffs_idx,size_t count);
extern void  sort_by_idxs(double* restrict o_coeffs,size_t* restrict o_coeffs_idx,size_t count);

#endif
