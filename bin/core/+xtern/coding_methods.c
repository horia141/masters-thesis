#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#include "acml/acml.h"
#include "gsl/gsl_statistics.h"
#include "gsl/gsl_multimin.h"
#include "gsl/gsl_rng.h"

#include "coding_methods.h"
#include "latools.h"

size_t
correlation_coding_tmps_length(
    size_t  geometry,
    size_t  word_count,
    size_t  coeff_count) {
    return word_count * sizeof(double) + // for "similarities".
           word_count * sizeof(size_t);  // for "similarities_idx".
}

size_t
matching_pursuit_coding_tmps_length(
    size_t  geometry,
    size_t  word_count,
    size_t  coeff_count) {
    return word_count * sizeof(double); // for "similarities".
}

size_t
orthogonal_matching_pursuit_coding_tmps_length(
    size_t  geometry,
    size_t  word_count,
    size_t  coeff_count) {
    return word_count * sizeof(double) +               // for "similarities".
           geometry * sizeof(double) +                 // for "residual".
           geometry * coeff_count * sizeof(double) +   // for "dict_transp_normalized".
           coeff_count * coeff_count * sizeof(double); // for "coeff_inversion_matrix".
}

size_t
optimized_orthogonal_matching_pursuit_coding_tmps_length(
    size_t  geometry,
    size_t  word_count,
    size_t  coeff_count) {
    return word_count * sizeof(bool) +                  // for "used_column_mask".
           geometry * sizeof(double) +                  // for "residual".
           geometry * word_count * sizeof(double) +     // for "dict_transp_tilde".
           geometry * coeff_count * sizeof(double) +    // for "dict_transp_normalized".
           coeff_count * coeff_count * sizeof(double) + // for "coeff_inversion_matrix".
           word_count * sizeof(double) +                // for "observation_column_tilde_dot".
           geometry * sizeof(double);                   // for "local_residual".
}

size_t
sparse_net_coding_tmps_length(
    size_t  geometry,
    size_t  word_count,
    size_t  coeff_count) {
    return word_count * sizeof(double) +  // for "initial_coeffs".
           word_count * sizeof(size_t) +  // for "initial_coeffs_idx".
           geometry * sizeof(double);     // for "local_coeffs".
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
    double* restrict  similarities;
    size_t* restrict  similarities_idx;

    curr_coding_tmps = (char* restrict)coding_tmps;

    similarities = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(double);
    similarities_idx = (size_t* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(size_t);

    dgemv('N',(int)word_count,(int)geometry,1,(double*)dict,(int)word_count,(double*)observation,1,0,similarities,1);
    fill_idx_1n(similarities_idx,word_count);

    sort_by_abs_coeffs(similarities,similarities_idx,word_count);

    memcpy(o_coeffs,similarities,coeff_count * sizeof(double));
    memcpy(o_coeffs_idx,similarities_idx,coeff_count * sizeof(size_t));
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
    double* restrict  similarities;
    size_t            max_idx;
    double            max_sim;
    size_t            ii;

    curr_coding_tmps = (char* restrict)coding_tmps;

    similarities = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(double);

    dgemv('N',(int)word_count,(int)geometry,1,(double*)dict,(int)word_count,(double*)observation,1,0,similarities,1);

    for (ii = 0; ii < coeff_count; ii++) {
        max_idx = (size_t)idamax((int)word_count,similarities,1) - 1;
        max_sim = similarities[max_idx];

        o_coeffs[ii] = max_sim;
        o_coeffs_idx[ii] = max_idx; /* There is a very very small chance of inserting a duplicate here. */

        daxpy((int)word_count,-max_sim,(double*)dict_x_dict_transp + max_idx * word_count,1,similarities,1);
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
    double* restrict        similarities;
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

    similarities = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(double);
    residual = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += geometry * sizeof(double);
    dict_transp_normalized = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += geometry * coeff_count * sizeof(double);
    coeff_inversion_matrix = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += coeff_count * coeff_count * sizeof(double);

    memcpy(residual,observation,geometry * sizeof(double));

    for (ii = 0; ii < coeff_count; ii++) {
        dgemv('N',(int)word_count,(int)geometry,1,(double*)dict,(int)word_count,(double*)residual,1,0,similarities,1);
        max_idx = (size_t)idamax((int)word_count,similarities,1) - 1;

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

void
optimized_orthogonal_matching_pursuit(
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
    bool* restrict          used_column_mask;
    double* restrict        residual;
    double* restrict        dict_transp_tilde;
    double* restrict        dict_transp_normalized;
    double* restrict        coeff_inversion_matrix;
    double* restrict        observation_column_tilde_dot;
    double* restrict        local_residual;
    double                  next_residual_norm;
    double                  min_next_residual;
    size_t                  min_idx;
    const double* restrict  winner_column_dict_transp;
    double* restrict        winner_column_dict_transp_tilde;
    double* restrict        curr_column_dict_transp_tilde;
    double* restrict        curr_column_dict_transp_normalized;
    double* restrict        curr_column_coeff_inversion_matrix;
    double                  column_tilde_proj_on_winner;
    double                  new_norm;
    size_t                  ii;
    size_t                  jj;

    curr_coding_tmps = (char* restrict)coding_tmps;

    used_column_mask = (bool* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(bool);
    residual = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += geometry * sizeof(double);
    dict_transp_tilde = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += geometry * word_count * sizeof(double);
    dict_transp_normalized = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += geometry * coeff_count * sizeof(double);
    coeff_inversion_matrix = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += coeff_count * coeff_count * sizeof(double);
    observation_column_tilde_dot = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(double);
    local_residual = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += geometry * sizeof(double);

    memset(used_column_mask,0,word_count * sizeof(bool));
    memcpy(residual,observation,geometry * sizeof(double));

    memcpy(dict_transp_tilde,dict_transp,geometry * word_count * sizeof(double));

    curr_column_dict_transp_tilde = dict_transp_tilde;

    for (ii = 0; ii < word_count; ii++) {
        new_norm = dnrm2((int)geometry,(double*)curr_column_dict_transp_tilde,1);
        dscal((int)geometry,1 / new_norm,(double*)curr_column_dict_transp_tilde,1);
        curr_column_dict_transp_tilde += geometry;
    }

    for (ii = 0; ii < coeff_count; ii++) {
        curr_column_dict_transp_tilde = dict_transp_tilde;

        min_next_residual = HUGE_VAL;
        min_idx = 0;

        for (jj = 0; jj < word_count; jj++) {
            if (!used_column_mask[jj]) {
                observation_column_tilde_dot[jj] = ddot((int)geometry,(double*)observation,1,(double*)curr_column_dict_transp_tilde,1);
                memcpy(local_residual,residual,geometry * sizeof(double));
                daxpy((int)geometry,-observation_column_tilde_dot[jj],(double*)curr_column_dict_transp_tilde,1,(double*)local_residual,1);
                next_residual_norm = dnrm2((int)geometry,(double*)local_residual,1);

                if (next_residual_norm < min_next_residual) {
                    min_next_residual = next_residual_norm;
                    min_idx = jj;
                }
            }

            curr_column_dict_transp_tilde += geometry;
        }

        winner_column_dict_transp = dict_transp + min_idx * geometry;
        winner_column_dict_transp_tilde = dict_transp_tilde + min_idx * geometry;
        curr_column_dict_transp_normalized = dict_transp_normalized + ii * geometry;
        curr_column_coeff_inversion_matrix = coeff_inversion_matrix + ii * coeff_count;

        dgemv('T',(int)geometry,(int)ii,1,(double*)dict_transp_normalized,(int)geometry,(double*)winner_column_dict_transp,1,0,(double*)curr_column_coeff_inversion_matrix,1);
        memcpy(curr_column_dict_transp_normalized,winner_column_dict_transp_tilde,geometry * sizeof(double));
        curr_column_coeff_inversion_matrix[ii] = ddot((int)geometry,(double*)curr_column_dict_transp_normalized,1,(double*)winner_column_dict_transp,1);

        used_column_mask[min_idx] = true;
        memset(winner_column_dict_transp_tilde,0,geometry * sizeof(double));

        o_coeffs[ii] = observation_column_tilde_dot[min_idx];
        o_coeffs_idx[ii] = min_idx;

        daxpy((int)geometry,-o_coeffs[ii],(double*)curr_column_dict_transp_normalized,1,(double*)residual,1);

        curr_column_dict_transp_tilde = dict_transp_tilde;

        for (jj = 0; jj < word_count; jj++) {
            if (!used_column_mask[jj]) {
                column_tilde_proj_on_winner = ddot((int)geometry,(double*)curr_column_dict_transp_tilde,1,(double*)curr_column_dict_transp_normalized,1);
                daxpy((int)geometry,-column_tilde_proj_on_winner,(double*)curr_column_dict_transp_normalized,1,(double*)curr_column_dict_transp_tilde,1);
                new_norm = dnrm2((int)geometry,(double*)curr_column_dict_transp_tilde,1);
                if (new_norm >= 1e-6) {
                    dscal((int)geometry,1 / new_norm,(double*)curr_column_dict_transp_tilde,1);
                }
            }

            curr_column_dict_transp_tilde += geometry;
        }
    }

    dtrsv('U','N','N',(int)coeff_count,(double*)coeff_inversion_matrix,(int)coeff_count,(double*)o_coeffs,1);
}

#define SPARSE_NET_INITIAL_STEP 0.001
#define SPARSE_NET_CG_TOL 0.01
#define SPARSE_NET_GRAD_TEST 0.001
#define SPARSE_NET_MAX_ITERS 50

struct _sparse_net_cost_params {
    size_t         geometry;
    size_t         word_count;
    const double*  dict;
    const double*  dict_transp;
    const double*  dict_x_dict_transp;
    const double*  observation;
    double         lambda;
    double         sigma;
    double*        local_coeffs;
};

static double
_sparse_net_cost_value(
    const gsl_vector*  current_coeffs,
    void*              params_t) {
    struct _sparse_net_cost_params*  params;
    double                           norm;
    double                           approx_term;
    double                           current_coeff_adj;
    double                           reg_term_sum;
    double                           reg_term;
    size_t                           ii;

    params = (struct _sparse_net_cost_params*)params_t;

    dgemv('N',(int)params->geometry,(int)params->word_count,1,(double*)params->dict_transp,(int)params->geometry,(double*)current_coeffs->data,1,0,params->local_coeffs,1);
    daxpy((int)params->geometry,-1,(double*)params->observation,1,params->local_coeffs,1);
    norm = dnrm2((int)params->geometry,(double*)params->local_coeffs,1);
    approx_term = 0.5 * norm * norm;

    reg_term_sum = 0;

    for (ii = 0; ii < params->word_count; ii++) {
        current_coeff_adj = current_coeffs->data[ii] / params->sigma;
        reg_term_sum += log(1 + current_coeff_adj * current_coeff_adj);
    }

    reg_term = params->lambda * reg_term_sum;

    return approx_term + reg_term;
}

static void
_sparse_net_cost_grad(
    const gsl_vector*  current_coeffs,
    void*              params_t,
    gsl_vector*        grad) {
    struct _sparse_net_cost_params*  params;
    double                           current_coeff_adj;
    size_t                           ii;

    params = (struct _sparse_net_cost_params*)params_t;

    dgemv('N',(int)params->word_count,(int)params->geometry,-1,(double*)params->dict,(int)params->word_count,(double*)params->observation,1,0,grad->data,1);
    dgemv('N',(int)params->word_count,(int)params->word_count,1,(double*)params->dict_x_dict_transp,(int)params->word_count,(double*)current_coeffs->data,1,1,grad->data,1);

    for (ii = 0; ii < params->word_count; ii++) {
	current_coeff_adj = current_coeffs->data[ii] / params->sigma;
	grad->data[ii] += (2 * params->lambda * current_coeff_adj) / (params->sigma * (1 + current_coeff_adj));
    }
}

static void
_sparse_net_cost_value_grad(
    const gsl_vector*  current_coeffs,
    void*              params,
    double*            f,
    gsl_vector*        grad) {
    *f = _sparse_net_cost_value(current_coeffs,params);
    _sparse_net_cost_grad(current_coeffs,params,grad);
}

void
sparse_net(
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
    char* restrict                  curr_coding_tmps;
    double* restrict                initial_coeffs;
    size_t* restrict                initial_coeffs_idx;
    double* restrict                local_coeffs;
    double                          lambda_sigma_ratio;
    double                          sigma;
    double                          lambda;
    gsl_rng*                        rnd_generator;
    gsl_block                       initial_coeffs_block;
    gsl_vector                      initial_coeffs_vector;
    struct _sparse_net_cost_params  cost_fn_params;
    gsl_multimin_function_fdf       cost_fn_desc;
    gsl_multimin_fdfminimizer*      minimizer;
    int                             minimization_status;
    size_t                          ii;

    curr_coding_tmps = (char* restrict)coding_tmps;
    
    initial_coeffs = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(double);
    initial_coeffs_idx = (size_t* restrict)curr_coding_tmps;
    curr_coding_tmps += word_count * sizeof(size_t);
    local_coeffs = (double* restrict)curr_coding_tmps;
    curr_coding_tmps += geometry * sizeof(double);

    lambda_sigma_ratio = *(double*)(((void**)coding_params)[0]);
    sigma = gsl_stats_sd(observation,1,geometry);
    lambda = lambda_sigma_ratio * sigma;

    rnd_generator = (gsl_rng*)(((void**)coding_params)[1]);

    for (ii = 0; ii < word_count; ii++) {
        initial_coeffs[ii] = 0.1 * gsl_rng_uniform(rnd_generator) - 0.05;
    }

    fill_idx_1n(initial_coeffs_idx,word_count);

    initial_coeffs_block.size = word_count;
    initial_coeffs_block.data = initial_coeffs;
    initial_coeffs_vector.size = word_count;
    initial_coeffs_vector.stride = 1;
    initial_coeffs_vector.data = initial_coeffs;
    initial_coeffs_vector.block = &initial_coeffs_block;
    initial_coeffs_vector.owner = 0;

    cost_fn_params.geometry = geometry;
    cost_fn_params.word_count = word_count;
    cost_fn_params.dict = dict;
    cost_fn_params.dict_transp = dict_transp;
    cost_fn_params.dict_x_dict_transp = dict_x_dict_transp;
    cost_fn_params.observation = observation;
    cost_fn_params.lambda = lambda;
    cost_fn_params.sigma = sigma;
    cost_fn_params.local_coeffs = local_coeffs;

    cost_fn_desc.n = word_count;
    cost_fn_desc.f = _sparse_net_cost_value;
    cost_fn_desc.df = _sparse_net_cost_grad;
    cost_fn_desc.fdf = _sparse_net_cost_value_grad;
    cost_fn_desc.params = &cost_fn_params;    

    minimizer = gsl_multimin_fdfminimizer_alloc(gsl_multimin_fdfminimizer_conjugate_fr,word_count);
    gsl_multimin_fdfminimizer_set(minimizer,&cost_fn_desc,&initial_coeffs_vector,SPARSE_NET_INITIAL_STEP,SPARSE_NET_CG_TOL);

    for (ii = 0; ii < SPARSE_NET_MAX_ITERS; ii++) {
        minimization_status = gsl_multimin_fdfminimizer_iterate(minimizer);

        if (minimization_status) {
            break;
        }

        if (gsl_multimin_test_gradient(minimizer->gradient,SPARSE_NET_GRAD_TEST) == GSL_SUCCESS) {
            break;
        }
    }

    memcpy(initial_coeffs,minimizer->x->data,word_count * sizeof(double));

    gsl_multimin_fdfminimizer_free(minimizer);

    sort_by_abs_coeffs(initial_coeffs,initial_coeffs_idx,word_count);

    memcpy(o_coeffs,initial_coeffs,coeff_count * sizeof(double));
    memcpy(o_coeffs_idx,initial_coeffs_idx,coeff_count * sizeof(size_t));
}
