#include <math.h>
#include <string.h>

#include "latools.h"

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
