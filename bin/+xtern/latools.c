#include <math.h>
#include <string.h>

#include "latools.h"

void
dict_obs_inner_product(
    double* restrict        o_similarities,
    size_t                  geometry,
    size_t                  word_count,
    const double* restrict  dict,
    const double* restrict  dict_transp,
    const double* restrict  observation) {
    double                  curr_result_0;
    const double* restrict  curr_dict_transp_word_0;
    double                  curr_result_1;
    const double* restrict  curr_dict_transp_word_1;
    double                  curr_result_2;
    const double* restrict  curr_dict_transp_word_2;
    double                  curr_result_3;
    const double* restrict  curr_dict_transp_word_3;
    size_t                  adjusted_word_count;
    size_t                  curr_word;
    size_t                  curr_component;

    adjusted_word_count = word_count - (word_count % 4);

    curr_dict_transp_word_0 = dict_transp + 0 * geometry;
    curr_dict_transp_word_1 = dict_transp + 1 * geometry;
    curr_dict_transp_word_2 = dict_transp + 2 * geometry;
    curr_dict_transp_word_3 = dict_transp + 3 * geometry;

    for (curr_word = 0; curr_word < adjusted_word_count; curr_word += 4) {
	curr_result_0 = 0;
	curr_result_1 = 0;
	curr_result_2 = 0;
	curr_result_3 = 0;

        for (curr_component = 0; curr_component < geometry; curr_component++) {
	    const double curr_observation = observation[curr_component];
	    const double curr_dict_transp_0 = curr_dict_transp_word_0[curr_component];
	    const double curr_dict_transp_1 = curr_dict_transp_word_1[curr_component];
	    const double curr_dict_transp_2 = curr_dict_transp_word_2[curr_component];
	    const double curr_dict_transp_3 = curr_dict_transp_word_3[curr_component];

	    curr_result_0 = curr_result_0 + curr_observation * curr_dict_transp_0;
	    curr_result_1 = curr_result_1 + curr_observation * curr_dict_transp_1;
	    curr_result_2 = curr_result_2 + curr_observation * curr_dict_transp_2;
	    curr_result_3 = curr_result_3 + curr_observation * curr_dict_transp_3;
        }

	o_similarities[curr_word + 0] = curr_result_0;
	o_similarities[curr_word + 1] = curr_result_1;
	o_similarities[curr_word + 2] = curr_result_2;
	o_similarities[curr_word + 3] = curr_result_3;
	curr_dict_transp_word_0 = curr_dict_transp_word_0 + 4 * geometry;
	curr_dict_transp_word_1 = curr_dict_transp_word_1 + 4 * geometry;
	curr_dict_transp_word_2 = curr_dict_transp_word_2 + 4 * geometry;
	curr_dict_transp_word_3 = curr_dict_transp_word_3 + 4 * geometry;
    }

    for (curr_word = adjusted_word_count; curr_word < word_count; curr_word++) { 
	curr_result_0 = 0;

        for (curr_component = 0; curr_component < geometry; curr_component++) {
	    const double curr_observation = observation[curr_component];
	    const double curr_dict_transp_0 = curr_dict_transp_word_0[curr_component];

	    curr_result_0 = curr_result_0 + curr_observation * curr_dict_transp_0;
        }

	o_similarities[curr_word] = curr_result_0;
	curr_dict_transp_word_0 = curr_dict_transp_word_0 + geometry;
    }
}

void
fill_idx_1n(
    size_t* restrict  o_idx,
    size_t            count) {
    size_t  ii;

    for (ii = 0; ii < count; ii++) {
        o_idx[ii] = ii;
    }
}

void
sort_by_abs_coeffs(
    double* restrict  o_coeffs,
    size_t* restrict  o_coeffs_idx,
    size_t            count) {
    size_t  median_idx;
    double  tmp_median;
    size_t  tmp_median_idx;
    double  fabs_o_coeffs_ii;
    double  fabs_o_coeffs_median_idx;
    size_t  ii;

    while (count > 1) {
        median_idx = 0;

        for (ii = 1; ii < count; ii++) {
	    fabs_o_coeffs_ii = fabs(o_coeffs[ii]);
	    fabs_o_coeffs_median_idx = fabs(o_coeffs[median_idx]);

            if ((fabs_o_coeffs_ii > fabs_o_coeffs_median_idx) || 
		((fabs_o_coeffs_ii == fabs_o_coeffs_median_idx) &&
		 (o_coeffs_idx[ii] < o_coeffs_idx[median_idx]))) {
		tmp_median = o_coeffs[median_idx];
		o_coeffs[median_idx] = o_coeffs[ii];
		o_coeffs[ii] = o_coeffs[median_idx + 1];
		o_coeffs[median_idx + 1] = tmp_median;
		tmp_median_idx = o_coeffs_idx[median_idx];
		o_coeffs_idx[median_idx] = o_coeffs_idx[ii];
		o_coeffs_idx[ii] = o_coeffs_idx[median_idx + 1];
		o_coeffs_idx[median_idx + 1] = tmp_median_idx;
                median_idx = median_idx + 1;
            }
        }

        sort_by_abs_coeffs(o_coeffs,o_coeffs_idx,median_idx);

	o_coeffs = o_coeffs + median_idx + 1;
	o_coeffs_idx = o_coeffs_idx + median_idx + 1;
	count = count - median_idx - 1;
    }

}

void
sort_by_idxs(
    double* restrict  o_coeffs,
    size_t* restrict  o_coeffs_idx,
    size_t            count) {
    size_t  median_idx;
    double  tmp_median;
    size_t  tmp_median_idx;
    size_t  ii;

    while (count > 1) {
        median_idx = 0;

        for (ii = 1; ii < count; ii++) {
            if (o_coeffs_idx[ii] < o_coeffs_idx[median_idx]) {
		tmp_median = o_coeffs[median_idx];
		o_coeffs[median_idx] = o_coeffs[ii];
		o_coeffs[ii] = o_coeffs[median_idx + 1];
		o_coeffs[median_idx + 1] = tmp_median;
		tmp_median_idx = o_coeffs_idx[median_idx];
		o_coeffs_idx[median_idx] = o_coeffs_idx[ii];
		o_coeffs_idx[ii] = o_coeffs_idx[median_idx + 1];
		o_coeffs_idx[median_idx + 1] = tmp_median_idx;
                median_idx = median_idx + 1;
            }
        }

        sort_by_idxs(o_coeffs,o_coeffs_idx,median_idx);

	o_coeffs = o_coeffs + median_idx + 1;
	o_coeffs_idx = o_coeffs_idx + median_idx + 1;
	count = count - median_idx - 1;
    }
}
