#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "acml.h"

#include "coding_methods.h"
#include "latools.h"

size_t
correlation_coding_tmps_length(
    size_t  geometry,
    size_t  word_count,
    size_t  coeff_count) {
    return word_count * sizeof(double) + word_count * sizeof(size_t);
}

size_t
matching_pursuit_coding_tmps_length(
    size_t  geometry,
    size_t  word_count,
    size_t  coeff_count) {
    return word_count * sizeof(double);
}

size_t
orthogonal_matching_pursuit_coding_tmps_length(
    size_t  geometry,
    size_t  word_count,
    size_t  coeff_count) {
    return word_count * sizeof(double) + geometry * sizeof(double) + geometry * coeff_count * sizeof(double) + coeff_count * coeff_count * sizeof(double);
}

void
correlation(
    double* restrict        o_coeffs,
    size_t* restrict        o_coeffs_idx,
    size_t                  geometry,
    size_t                  word_count,
    const double* restrict  dict,
    const double* restrict  dict_transp,
    const double* restrict  dict_x_dict_transp,
    size_t                  coeff_count,
    const void* restrict    coding_params,
    const double* restrict  observation,
    void* restrict          coding_tmps) {
    char* restrict    curr_coding_tmps;
    double* restrict  tmp_similarities;
    size_t* restrict  tmp_similarities_idx;

    curr_coding_tmps = (char* restrict)coding_tmps;

    tmp_similarities = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(double);
    tmp_similarities_idx = (size_t* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(size_t);

    dgemv('N',(int)word_count,(int)geometry,1,(double*)dict,(int)word_count,(double*)observation,1,0,tmp_similarities,1);
    fill_idx_1n(tmp_similarities_idx,word_count);

    sort_by_abs_coeffs(tmp_similarities,tmp_similarities_idx,word_count);

    memcpy(o_coeffs,tmp_similarities,coeff_count * sizeof(double));
    memcpy(o_coeffs_idx,tmp_similarities_idx,coeff_count * sizeof(size_t));
}

void
matching_pursuit(
    double* restrict        o_coeffs,
    size_t* restrict        o_coeffs_idx,
    size_t                  geometry,
    size_t                  word_count,
    const double* restrict  dict,
    const double* restrict  dict_transp,
    const double* restrict  dict_x_dict_transp,
    size_t                  coeff_count,
    const void* restrict    coding_params,
    const double* restrict  observation,
    void* restrict          coding_tmps) {
    char* restrict    curr_coding_tmps;
    double* restrict  tmp_similarities;
    size_t            max_idx;
    double            max_sim;
    size_t            ii;

    curr_coding_tmps = (char* restrict)coding_tmps;

    tmp_similarities = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(double);

    dgemv('N',(int)word_count,(int)geometry,1,(double*)dict,(int)word_count,(double*)observation,1,0,tmp_similarities,1);

    for (ii = 0; ii < coeff_count; ii++) {
        max_idx = (size_t)idamax((int)word_count,tmp_similarities,1) - 1;
        max_sim = tmp_similarities[max_idx];

        o_coeffs[ii] = max_sim;
        o_coeffs_idx[ii] = max_idx; /* There is a very very small chance of inserting a duplicate here. */

        daxpy((int)word_count,-max_sim,(double*)dict_x_dict_transp + max_idx * word_count,1,tmp_similarities,1);
    }
}

void
orthogonal_matching_pursuit(
    double* restrict        o_coeffs,
    size_t* restrict        o_coeffs_idx,
    size_t                  geometry,
    size_t                  word_count,
    const double* restrict  dict,
    const double* restrict  dict_transp,
    const double* restrict  dict_x_dict_transp,
    size_t                  coeff_count,
    const void* restrict    coding_params,
    const double* restrict  observation,
    void* restrict          coding_tmps) {
    char* restrict          curr_coding_tmps;
    double* restrict        tmp_similarities;
    double* restrict        residual;
    double* restrict        dict_transp_normalized;
    double* restrict        coeff_inversion_matrix;
    size_t                  max_idx;
    const double* restrict  winner_column_dict_transp;
    double* restrict        curr_column_dict_transp_normalized;
    double* restrict        curr_column_coeff_inversion_matrix;
    double                  new_norm;
    size_t                  ii;

    curr_coding_tmps = (char* restrict)coding_tmps;

    tmp_similarities = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(double);
    residual = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += geometry * sizeof(double);
    dict_transp_normalized = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += geometry * coeff_count * sizeof(double);
    coeff_inversion_matrix = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += coeff_count * coeff_count * sizeof(double);

    memcpy(residual,observation,geometry * sizeof(double));

    for (ii = 0; ii < coeff_count; ii++) {
        dgemv('N',(int)word_count,(int)geometry,1,(double*)dict,(int)word_count,(double*)residual,1,0,tmp_similarities,1);
        max_idx = (size_t)idamax((int)word_count,tmp_similarities,1) - 1;

	winner_column_dict_transp = dict_transp + max_idx * geometry;
	curr_column_dict_transp_normalized = dict_transp_normalized + ii * geometry;
	curr_column_coeff_inversion_matrix = coeff_inversion_matrix + ii * coeff_count;

        dgemv('T',(int)geometry,(int)ii,1,(double*)dict_transp_normalized,(int)geometry,(double*)winner_column_dict_transp,1,0,(double*)curr_column_coeff_inversion_matrix,1);
        memcpy(curr_column_dict_transp_normalized,winner_column_dict_transp,geometry * sizeof(double));
        dgemv('N',(int)geometry,(int)ii,-1,(double*)dict_transp_normalized,(int)geometry,(double*)curr_column_coeff_inversion_matrix,1,1,(double*)curr_column_dict_transp_normalized,1);
        new_norm = dnrm2((int)geometry,(double*)curr_column_dict_transp_normalized,1);
        dscal((int)geometry,1 / new_norm,(double*)curr_column_dict_transp_normalized,1);
        curr_column_coeff_inversion_matrix[ii] = ddot((int)geometry,(double*)curr_column_dict_transp_normalized,1,(double*)winner_column_dict_transp,1);

        o_coeffs[ii] = ddot((int)geometry,(double*)curr_column_dict_transp_normalized,1,(double*)observation,1);
        o_coeffs_idx[ii] = max_idx;

        daxpy((int)geometry,-o_coeffs[ii],(double*)curr_column_dict_transp_normalized,1,(double*)residual,1);
    }

    dtrsv('U','N','N',(int)coeff_count,(double*)coeff_inversion_matrix,(int)coeff_count,(double*)o_coeffs,1);
}
