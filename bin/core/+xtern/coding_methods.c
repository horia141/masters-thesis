#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "acml.h"

#include "coding_methods.h"
#include "latools.h"

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
    double* restrict  tmp_similarities;
    size_t* restrict  tmp_similarities_idx;

    tmp_similarities = (double* restrict)coding_tmps;
    tmp_similarities_idx = (size_t* restrict)((double* restrict)coding_tmps + word_count);

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
    double* restrict  tmp_similarities;
    size_t            max_idx;
    double            max_abs_sim;
    double            max_sim;
    double            test_abs_sim;
    size_t            ii;
    size_t            jj;

    tmp_similarities = (double* restrict)coding_tmps;

    dgemv('N',(int)word_count,(int)geometry,1,(double*)dict,(int)word_count,(double*)observation,1,0,tmp_similarities,1);

    for (ii = 0; ii < coeff_count; ii++) {
	max_idx = 0;
	max_abs_sim = -HUGE_VAL;

	for (jj = 0; jj < word_count; jj++) {
	    test_abs_sim = fabs(tmp_similarities[jj]);

	    if (test_abs_sim > max_abs_sim) {
		max_idx = jj;
		max_abs_sim = test_abs_sim;
	    }
	}

	max_sim = tmp_similarities[max_idx];

	o_coeffs[ii] = tmp_similarities[max_idx];
	o_coeffs_idx[ii] = max_idx; /* There is a very very small chance of inserting a duplicate here. */

	for (jj = 0; jj < word_count; jj++) {
	    tmp_similarities[jj] -= max_sim * dict_x_dict_transp[max_idx * word_count + jj];
	}
    }
}
