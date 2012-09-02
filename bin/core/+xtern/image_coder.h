#ifndef _IMAGE_CODER_H
#define _IMAGE_CODER_H

#include "base_defines.h"

enum coding_type {
    CORRELATION,
    MATCHING_PURSUIT,
    ORTHOGONAL_MATCHING_PURSUIT
};

enum nonlinear_type {
    LINEAR,
    LOGISTIC,
    GLOBAL_ORDER
};

enum polarity_split_type {
    NONE,
    NO_SIGN,
    KEEP_SIGN
};

enum reduce_type {
    SUBSAMPLE,
    MAX_NO_SIGN,
    MAX_KEEP_SIGN,
    SUM_ABS,
    SUM_SQR
};

extern size_t  code_image_new_geometry(size_t row_count,size_t col_count,size_t word_count,enum polarity_split_type polarity_split_type,size_t reduce_spread);
extern size_t  code_image_coding_tmps_length(size_t row_count,size_t col_count,size_t patch_row_count,size_t patch_col_count,enum coding_type coding_type,size_t word_count,size_t coeff_count,size_t reduce_spread);
extern void    code_image(size_t* restrict o_coeff_count,double* restrict o_coeffs,size_t* restrict o_coeffs_idx,size_t geometry,size_t row_count,size_t col_count,size_t patch_row_count,size_t patch_col_count,enum coding_type coding_type,size_t word_count,const double* restrict dict,const double* restrict dict_transp,const double* restrict dict_x_dict_transp,size_t coeff_count,const void* restrict coding_param,enum nonlinear_type nonlinear_type,const double* restrict nonlinear_modulator,enum polarity_split_type polarity_split_type,enum reduce_type reduce_type,size_t reduce_spread,const double* restrict observation,void* restrict coding_tmps);

#endif
