#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "image_coder.h"
#include "coding_methods.h"
#include "latools.h"

typedef double (*reduce_function)(double,double);

static double
_max_no_sign(
    double  current,
    double  new_value) {
    double  fnew_value;

    fnew_value = fabs(new_value);

    if (current < fnew_value) {
	return fnew_value;
    } else {
	return current;
    }
}

static double
_max_keep_sign(
    double  current,
    double  new_value) {
    if (fabs(current) < fabs(new_value)) {
	return new_value;
    } else {
	return current;
    }
}

static double
_sum_abs(
    double  current,
    double  new_value) {
    return current + fabs(new_value);
}

static double
_sum_sqr(
    double  current,
    double  new_value) {
    return current + new_value * new_value;
}

size_t
code_image_new_geometry(
    size_t                    row_count,
    size_t                    col_count,
    enum resize_type          resize_type,
    size_t                    new_row_count,
    size_t                    new_col_count,
    size_t                    word_count,
    enum polarity_split_type  polarity_split_type,
    size_t                    reduce_spread) {
    size_t  aftresize_row_count;
    size_t  aftresize_col_count;
    size_t  aftcoding_row_count;
    size_t  aftcoding_col_count;
    size_t  polarity_split_multiplier;
    size_t  aftreduce_row_count;
    size_t  aftreduce_col_count;

    if (resize_type == IDENTITY) {
	aftresize_row_count = row_count;
	aftresize_col_count = col_count;
    } else if (resize_type == CLOSEST) {
	aftresize_row_count = new_row_count;
	aftresize_col_count = new_col_count;
    } else {
    }

    aftcoding_row_count = aftresize_row_count - (aftresize_row_count % reduce_spread);
    aftcoding_col_count = aftresize_col_count - (aftresize_col_count % reduce_spread);

    if (polarity_split_type == NONE) {
	polarity_split_multiplier = 1;
    } else if (polarity_split_type == NO_SIGN || polarity_split_type == KEEP_SIGN) {
	polarity_split_multiplier = 2;
    } else {
    }

    aftreduce_row_count = aftcoding_row_count / reduce_spread;
    aftreduce_col_count = aftcoding_col_count / reduce_spread;

    return polarity_split_multiplier * aftreduce_row_count * aftreduce_col_count * word_count;
}

size_t
code_image_coding_tmps_length(
    size_t row_count,
    size_t col_count,
    size_t patch_row_count,
    size_t patch_col_count,
    enum resize_type resize_type,
    size_t new_row_count,
    size_t new_col_count,
    enum coding_type coding_type,
    size_t word_count,
    size_t coeff_count,
    size_t reduce_spread) {
    size_t  coding_tmps_length;
    size_t  aftresize_row_count;
    size_t  aftresize_col_count;
    size_t  aftcoding_row_count;
    size_t  aftcoding_col_count;

    coding_tmps_length = 0;

    if (resize_type == IDENTITY) {
	aftresize_row_count = row_count;
	aftresize_col_count = col_count;
	coding_tmps_length += 0;
    } else if (resize_type == CLOSEST) {
	aftresize_row_count = new_row_count;
	aftresize_col_count = new_col_count;
	coding_tmps_length += aftresize_row_count * aftresize_col_count * sizeof(double);
    } else {
    }

    aftcoding_row_count = aftresize_row_count - (aftresize_row_count % reduce_spread);
    aftcoding_col_count = aftresize_col_count - (aftresize_col_count % reduce_spread);

    coding_tmps_length += patch_row_count * patch_col_count * sizeof(double);
    coding_tmps_length += coeff_count * aftcoding_row_count * aftcoding_col_count * sizeof(double);
    coding_tmps_length += coeff_count * aftcoding_row_count * aftcoding_col_count * sizeof(size_t);

    if (coding_type == CORRELATION) {
	coding_tmps_length += word_count * sizeof(double) + word_count * sizeof(size_t);
    } else if (coding_type == CORRELATION_ORDER) {
	coding_tmps_length += word_count * sizeof(double) + word_count * sizeof(size_t);
    } else if (coding_type == MATCHING_PURSUIT) {
	coding_tmps_length += word_count * sizeof(double);
    } else {
    }

    coding_tmps_length += reduce_spread * reduce_spread * sizeof(size_t);

    return coding_tmps_length;
}

void
code_image(
    size_t* restrict          o_coeffs_count,
    double* restrict          o_coeffs,
    size_t* restrict          o_coeffs_idx,
    size_t                    geometry,
    size_t                    row_count,
    size_t                    col_count,
    size_t                    patch_row_count,
    size_t                    patch_col_count,
    enum resize_type          resize_type,
    size_t                    new_row_count,
    size_t                    new_col_count,
    enum coding_type          coding_type,
    size_t                    word_count,
    const double* restrict    dict,
    const double* restrict    dict_transp,
    const double* restrict    dict_x_dict_transp,
    size_t                    coeff_count,
    const void* restrict      coding_params,
    enum nonlinear_type       nonlinear_type,
    enum polarity_split_type  polarity_split_type,
    enum reduce_type          reduce_type,
    size_t                    reduce_spread,
    const double* restrict    observation,
    void* restrict            coding_tmps) {
    size_t                  aftresize_row_count;
    size_t                  aftresize_col_count;
    const double* restrict  resized_observation;
    char* restrict          curr_coding_tmps;
    size_t                  aftcoding_row_count;
    size_t                  aftcoding_col_count;
    double* restrict        coded_patches;
    size_t* restrict        coded_patches_idx;
    size_t                  polarity_split_multiplier;
    size_t                  aftreduce_row_count;
    size_t                  aftreduce_col_count;

    curr_coding_tmps = (char* restrict)coding_tmps;

    /* Resizing layer. */

    if (resize_type == IDENTITY) {
	/* Nothing to do so just build stage output. */

	aftresize_row_count = row_count;
	aftresize_col_count = col_count;
	resized_observation = observation;
    } else if (resize_type == CLOSEST) {
	size_t                  row_skip;
	size_t                  col_skip;
	size_t                  actual_row_count;
	size_t                  actual_col_count;
	double* restrict        resized_observation_t;
	double* restrict        resized_observation_ptr;
	const double* restrict  curr_observation_col;
	size_t                  cc;
	size_t                  rr;

	/* Make a resized version of the orginal image. */

	aftresize_row_count = new_row_count;
	aftresize_col_count = new_col_count;

	row_skip = row_count / new_row_count;
	col_skip = col_count / new_col_count;

	actual_row_count = row_count - (row_count % row_skip);
	actual_col_count = col_count - (col_count % col_skip);

	resized_observation_t = (double* restrict)coding_tmps;
	curr_coding_tmps += aftresize_row_count * aftresize_col_count * sizeof(double);
	resized_observation_ptr = resized_observation_t;
	curr_observation_col = observation;
	resized_observation = resized_observation_t;

	for (cc = 0; cc < actual_col_count; cc += col_skip) {
	    for (rr = 0; rr < actual_row_count; rr += row_skip) {
		*resized_observation_ptr = *(curr_observation_col + rr);
		resized_observation_ptr++;
	    }

	    curr_observation_col += row_count * col_skip;
	}
    } else {
    }

    /* Coding layer. */

    {
	double* restrict         patch_for_coding;
	double* restrict         patch_for_coding_ptr;
	const double*  restrict  curr_observation_col;
	char* restrict           coder_coding_tmps;
	coding_method_t          coding_method;
	size_t                   patch_side_row;
	size_t                   patch_init_row;
	size_t                   patch_final_row;
	size_t                   patch_skipped_initial_rows;
	size_t                   patch_skipped_final_rows;
	size_t                   patch_side_col;
	size_t                   patch_init_col;
	size_t                   patch_final_col;
	size_t                   patch_skipped_initial_cols;
	size_t                   curr_patch_offset;
	size_t                   cc;
	size_t                   rr;
	size_t                   rr_1;
	size_t                   cc_1;

	aftcoding_row_count = aftresize_row_count - (aftresize_row_count % reduce_spread);
	aftcoding_col_count = aftresize_col_count - (aftresize_col_count % reduce_spread);

	patch_for_coding = (double* restrict)curr_coding_tmps;
	curr_coding_tmps += patch_row_count * patch_col_count * sizeof(double);
	coded_patches = (double* restrict)curr_coding_tmps;
	curr_coding_tmps += coeff_count * aftcoding_row_count * aftcoding_col_count * sizeof(double);
	coded_patches_idx = (size_t* restrict)curr_coding_tmps;
	curr_coding_tmps += coeff_count * aftcoding_row_count * aftcoding_col_count * sizeof(size_t);

	if (coding_type == CORRELATION) {
	    coding_method = correlation;
	    coder_coding_tmps = curr_coding_tmps;
	    curr_coding_tmps += word_count * sizeof(double) + word_count * sizeof(size_t);
	} else if (coding_type == CORRELATION_ORDER) {
	    coding_method = correlation_order;
	    coder_coding_tmps = curr_coding_tmps;
	    curr_coding_tmps += word_count * sizeof(double) + word_count * sizeof(size_t);
	} else if (coding_type == MATCHING_PURSUIT) {
	    coding_method = matching_pursuit;
	    coder_coding_tmps = curr_coding_tmps;
	    curr_coding_tmps += word_count * sizeof(double);
	} else {
	}

	for (cc = 0; cc < aftcoding_col_count; cc++) {
	    for (rr = 0; rr < aftcoding_row_count; rr++) {
		/* Copy patch to temporary storage before actual coding. */

		/* We're using "aftresize_(row|col)_count" here becase we can allow
		   patches to extend in the whole image, not just the pixels in the
		   reduce areas. */

		patch_side_row = (patch_row_count - 1) / 2;
		patch_init_row = rr < patch_side_row ? 0 : (rr - patch_side_row);
		patch_final_row = aftresize_row_count < (rr + patch_side_row + 1) ? aftresize_row_count : (rr + patch_side_row + 1);
		patch_skipped_initial_rows = rr < patch_side_row ? (patch_side_row - rr) : 0;
		patch_skipped_final_rows = aftresize_row_count < (rr + patch_side_row + 1) ? (rr + patch_side_row + 1 - aftresize_row_count) : 0;

		patch_side_col = (patch_col_count - 1) / 2;
		patch_init_col = cc < patch_side_col ? 0 : (cc - patch_side_col);
		patch_final_col = aftresize_col_count < (cc + patch_side_col + 1) ? aftresize_col_count : (cc + patch_side_col + 1);
		patch_skipped_initial_cols = cc < patch_side_col ? (patch_side_col - cc) : 0;

		memset(patch_for_coding,0,patch_row_count * patch_col_count * sizeof(double));

		patch_for_coding_ptr = patch_for_coding + patch_skipped_initial_cols * patch_row_count + patch_skipped_initial_rows;
		curr_observation_col = resized_observation + patch_init_col * aftresize_row_count;

		for (cc_1 = patch_init_col; cc_1 < patch_final_col; cc_1++) {
		    for (rr_1 = patch_init_row; rr_1 < patch_final_row; rr_1++) {
			*patch_for_coding_ptr = *(curr_observation_col + rr_1);
			patch_for_coding_ptr++;
		    }

		    patch_for_coding_ptr += patch_skipped_initial_rows + patch_skipped_final_rows;
		    curr_observation_col += aftresize_row_count;
		}

		/* Perform actual coding. */

		curr_patch_offset = (cc / reduce_spread) * (aftresize_row_count / reduce_spread) * (reduce_spread * reduce_spread) +
		                    (rr / reduce_spread) * (reduce_spread * reduce_spread) +
		                    (cc % reduce_spread) * reduce_spread + (rr % reduce_spread);

		coding_method(coded_patches + curr_patch_offset * coeff_count,coded_patches_idx + curr_patch_offset * coeff_count,
			      patch_row_count * patch_col_count,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,
			      coding_params,patch_for_coding,coder_coding_tmps);
	    }
	}
    }

    /* Nonlinear layer. */

    if (nonlinear_type == LINEAR) {
    } else if (nonlinear_type == LOGISTIC) {
	size_t  ii;

    	for (ii = 0; ii < aftcoding_row_count * aftcoding_col_count * coeff_count; ii++) {
    	    coded_patches[ii] = 1 / (1 + exp(-coded_patches[ii])) - 0.5;
    	}
    } else {
    }

    /* Polarity split layer. */

    if (polarity_split_type == NONE) {
	polarity_split_multiplier = 1;
    } else if (polarity_split_type == NO_SIGN || polarity_split_type == KEEP_SIGN) {
	double* restrict  coded_patches_ptr;
	size_t* restrict  coded_patches_idx_ptr;
	double            polarity_multiplier;
	size_t            ii;
	size_t            jj;

    	coded_patches_ptr = coded_patches;
    	coded_patches_idx_ptr = coded_patches_idx;

    	if (polarity_split_type == NO_SIGN) {
    	    polarity_multiplier = -1;
    	} else {
    	    polarity_multiplier = 1;
    	}

    	for (ii = 0; ii < aftcoding_row_count * aftcoding_col_count; ii++) {
    	    for (jj = 0; jj < coeff_count; jj++,coded_patches_ptr++,coded_patches_idx_ptr++) {
    		if (*coded_patches_ptr < 0) {
    		    *coded_patches_ptr = polarity_multiplier * *coded_patches_ptr;
    		    *coded_patches_idx_ptr += word_count;
    		}
    	    }

    	    sort_by_idxs(coded_patches_ptr - coeff_count,coded_patches_idx_ptr - coeff_count,coeff_count);
    	}

	polarity_split_multiplier = 2;
    } else {
    }

    /* Reduce layer - HERE BE DRAGONS. */

    {
	size_t            reduce_spread_2;
	size_t* restrict  curr_indices;
	double* restrict  coded_patches_ptr;
	size_t* restrict  coded_patches_idx_ptr;
	double            final_result;
	bool              set_final_result;
	size_t            coeffs_count;
	reduce_function   reduce_function;
	size_t            ii;
	size_t            jj;
	size_t            kk;

	aftreduce_row_count = aftcoding_row_count / reduce_spread;
	aftreduce_col_count = aftcoding_col_count / reduce_spread;

	reduce_spread_2 = reduce_spread * reduce_spread;

	curr_indices = (size_t* restrict)curr_coding_tmps;
	curr_coding_tmps += reduce_spread_2 * sizeof(size_t);

	coded_patches_ptr = coded_patches;
	coded_patches_idx_ptr = coded_patches_idx;

	coeffs_count = 0;

	if (reduce_type == SUBSAMPLE) {
	    for (ii = 0; ii < aftreduce_row_count * aftreduce_col_count; ii++) {
		for (jj = 0; jj < coeff_count; jj++) {
		    o_coeffs[coeffs_count] = coded_patches_ptr[jj];
		    o_coeffs_idx[coeffs_count] = ii + coded_patches_idx_ptr[jj] * aftreduce_row_count * aftreduce_col_count;
		    coeffs_count++;
		}

		coded_patches_ptr += reduce_spread * reduce_spread * coeff_count;
		coded_patches_idx_ptr += reduce_spread * reduce_spread * coeff_count;
	    }

	    sort_by_idxs(o_coeffs,o_coeffs_idx,coeffs_count);
	    *o_coeffs_count = coeffs_count;
	} else {
	    if (reduce_type == MAX_NO_SIGN) {
		reduce_function = _max_no_sign;
	    } else if (reduce_type == MAX_KEEP_SIGN) {
		reduce_function = _max_keep_sign;
	    } else if (reduce_type == SUM_ABS) {
		reduce_function = _sum_abs;
	    } else if (reduce_type == SUM_SQR) {
		reduce_function = _sum_sqr;
	    } else {
	    }
	    
	    for (ii = 0; ii < aftreduce_row_count * aftreduce_col_count; ii++) {
		memset(curr_indices,0,reduce_spread_2 * sizeof(size_t));

		for (jj = 0; jj < word_count * polarity_split_multiplier; jj++) {
		    final_result = 0;
		    set_final_result = false;

		    for (kk = 0; kk < reduce_spread_2; kk++) {
			if (curr_indices[kk] < coeff_count && (coded_patches_idx_ptr[kk * coeff_count + curr_indices[kk]] == jj)) {
			    final_result = reduce_function(final_result,coded_patches_ptr[kk * coeff_count + curr_indices[kk]]);
			    curr_indices[kk]++;
			    set_final_result = true;
			}
		    }

		    if (set_final_result && final_result != 0) {
			o_coeffs[coeffs_count] = final_result;
			o_coeffs_idx[coeffs_count] = ii + jj * aftreduce_row_count * aftreduce_col_count;
			coeffs_count++;
		    }
		}

		coded_patches_ptr += reduce_spread * reduce_spread * coeff_count;
		coded_patches_idx_ptr += reduce_spread * reduce_spread * coeff_count;
	    }

	    sort_by_idxs(o_coeffs,o_coeffs_idx,coeffs_count);
	    *o_coeffs_count = coeffs_count;
	}
    }
}
