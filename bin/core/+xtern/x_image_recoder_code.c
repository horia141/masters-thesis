#include <stdlib.h>
#include <string.h>

#include <pthread.h>

#include "mex.h"

#include "gsl/gsl_rng.h"

#include "x_mex_interface.h"
#include "base_defines.h"
#include "task_control.h"
#include "image_coder.h"

enum output_decoder {
    O_SAMPLE_CODED       = 0,
    O_OBSERVATIONS_PERM  = 1,
    OUTPUTS_COUNT
};

enum input_decoder {
    I_ROW_COUNT            = 0,
    I_COL_COUNT            = 1,
    I_PATCH_ROW_COUNT      = 2,
    I_PATCH_COL_COUNT      = 3,
    I_CODING_TYPE          = 4,
    I_DICT                 = 5,
    I_DICT_TRANSP          = 6,
    I_DICT_X_DICT_TRANSP   = 7,
    I_COEFF_COUNT          = 8,
    I_CODING_PARAMS        = 9,
    I_NONLINEAR_TYPE       = 10,
    I_NONLINEAR_MODULATOR  = 11,
    I_POLARITY_SPLIT_TYPE  = 12,
    I_REDUCE_TYPE          = 13,
    I_REDUCE_SPREAD        = 14,
    I_SAMPLE               = 15,
    I_NUM_WORKERS          = 16,
    INPUTS_COUNT
};

struct global_info {
    size_t                    geometry;
    size_t                    new_geometry;
    size_t                    row_count;
    size_t                    col_count;
    size_t                    patch_row_count;
    size_t                    patch_col_count;
    enum coding_type          coding_type;
    size_t                    word_count;
    const double*             dict;
    const double*             dict_transp;
    const double*             dict_x_dict_transp;
    size_t                    coeff_count;
    const void*               coding_params;
    enum nonlinear_type       nonlinear_type;
    const double*             nonlinear_modulator;
    enum polarity_split_type  polarity_split_type;
    enum reduce_type          reduce_type;
    size_t                    reduce_spread;
};

struct global_vars {
    double*          o_sample_coded_pr;
    size_t*          o_sample_coded_ir;
    size_t*          o_sample_coded_jc;
    size_t*          o_observations_perm;
    size_t           current_sample_coded_count;
    size_t           current_sample_coded_length;
    pthread_mutex_t  coeffs_queue_control;
};

struct task_info {
    size_t         observation_id;
    const double*  observation;
};

static void
do_task(
    size_t                     id,
    const struct global_info*  global_info,
    struct global_vars*        global_vars,
    size_t                     task_info_count,
    struct task_info*          task_info) {
    size_t   o_coeffs_count;
    double*  o_coeffs;
    size_t*  o_coeffs_idx;
    char*    coding_tmps;
    size_t   initial_sample_coded_count;
    size_t   initial_sample_coded_length;
    void*    param_table[2] = {NULL,NULL};
    size_t   ii;

    o_coeffs = (double*)malloc(global_info->new_geometry * sizeof(double));
    o_coeffs_idx = (size_t*)malloc(global_info->new_geometry * sizeof(double));
    coding_tmps = (char*)malloc(code_image_coding_tmps_length(global_info->row_count,global_info->col_count,global_info->patch_row_count,global_info->patch_col_count,
							      global_info->coding_type,global_info->word_count,global_info->coeff_count,global_info->reduce_spread));

    if (global_info->coding_type == SPARSE_NET) {
	double*   local_lambda_sigma_ratio;
	gsl_rng*  rnd_generator;

	local_lambda_sigma_ratio = (double*)malloc(sizeof(double));
	*local_lambda_sigma_ratio = ((double*)global_info->coding_params)[0];

	rnd_generator = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rnd_generator,id);

	param_table[0] = local_lambda_sigma_ratio;
	param_table[1] = rnd_generator;
    }

    for (ii = 0; ii < task_info_count; ii++) {
	code_image(&o_coeffs_count,o_coeffs,o_coeffs_idx,
		   global_info->geometry,global_info->row_count,global_info->col_count,
		   global_info->patch_row_count,global_info->patch_col_count,
		   global_info->coding_type,global_info->word_count,global_info->dict,global_info->dict_transp,global_info->dict_x_dict_transp,global_info->coeff_count,&param_table,
		   global_info->nonlinear_type,global_info->nonlinear_modulator,global_info->polarity_split_type,global_info->reduce_type,global_info->reduce_spread,
		   task_info[ii].observation,coding_tmps);

	pthread_mutex_lock(&global_vars->coeffs_queue_control);
	initial_sample_coded_count = global_vars->current_sample_coded_count;
	initial_sample_coded_length = global_vars->current_sample_coded_length;
	global_vars->current_sample_coded_count += 1;
	global_vars->current_sample_coded_length += o_coeffs_count;
	pthread_mutex_unlock(&global_vars->coeffs_queue_control);

	memcpy(global_vars->o_sample_coded_pr + initial_sample_coded_length,o_coeffs,o_coeffs_count * sizeof(double));
	memcpy(global_vars->o_sample_coded_ir + initial_sample_coded_length,o_coeffs_idx,o_coeffs_count * sizeof(size_t));
	global_vars->o_sample_coded_jc[initial_sample_coded_count] = initial_sample_coded_length;
	global_vars->o_observations_perm[initial_sample_coded_count] = task_info[ii].observation_id;
    }

    if (global_info->coding_type == SPARSE_NET) {
	gsl_rng_free((gsl_rng*)param_table[1]);
	free((double*)param_table[1]);
    }

    free(coding_tmps);
    free(o_coeffs_idx);
    free(o_coeffs);
}

void
mexFunction(
    int             output_count,
    mxArray*        output[],
    int             input_count,
    const mxArray*  input[]) {
    size_t                    geometry;
    size_t                    row_count;
    size_t                    col_count;
    size_t                    patch_row_count;
    size_t                    patch_col_count;
    enum coding_type          coding_type;
    size_t                    word_count;
    const double*             dict;
    const double*             dict_transp;
    const double*             dict_x_dict_transp;
    size_t                    coeff_count;
    const void*               coding_params;
    enum nonlinear_type       nonlinear_type;
    const double*             nonlinear_modulator;
    enum polarity_split_type  polarity_split_type;
    enum reduce_type          reduce_type;
    size_t                    reduce_spread;
    size_t                    sample_count;
    const double*             sample;
    size_t                    num_workers;
    struct global_info        global_info;
    struct global_vars        global_vars;
    struct task_info*         task_info;
    int                       pthread_res;
    double*                   o_observations_perm;
    size_t                    ii;

    /* Extract relevant information from all inputs. */

    geometry = mxGetM(input[I_SAMPLE]);
    row_count = (size_t)mxGetScalar(input[I_ROW_COUNT]);
    col_count = (size_t)mxGetScalar(input[I_COL_COUNT]);
    patch_row_count = (size_t)mxGetScalar(input[I_PATCH_ROW_COUNT]);
    patch_col_count = (size_t)mxGetScalar(input[I_PATCH_COL_COUNT]);
    coding_type = (enum coding_type)mxGetScalar(input[I_CODING_TYPE]);
    word_count = mxGetM(input[I_DICT]);
    dict = mxGetPr(input[I_DICT]);
    dict_transp = mxGetPr(input[I_DICT_TRANSP]);
    dict_x_dict_transp = mxGetPr(input[I_DICT_X_DICT_TRANSP]);
    coeff_count = (size_t)mxGetScalar(input[I_COEFF_COUNT]);
    coding_params = mxGetPr(input[I_CODING_PARAMS]);
    nonlinear_type = (enum nonlinear_type)mxGetScalar(input[I_NONLINEAR_TYPE]);
    nonlinear_modulator = mxGetPr(input[I_NONLINEAR_MODULATOR]);
    polarity_split_type = (enum polarity_split_type)mxGetScalar(input[I_POLARITY_SPLIT_TYPE]);
    reduce_type = (enum reduce_type)mxGetScalar(input[I_REDUCE_TYPE]);
    reduce_spread = (size_t)mxGetScalar(input[I_REDUCE_SPREAD]);
    sample_count = mxGetN(input[I_SAMPLE]);
    sample = mxGetPr(input[I_SAMPLE]);
    num_workers = (size_t)mxGetScalar(input[I_NUM_WORKERS]);

    /* Build task distribution information. */

    global_info.geometry = geometry;
    global_info.new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);
    global_info.row_count = row_count;
    global_info.col_count = col_count;
    global_info.patch_row_count = patch_row_count;
    global_info.patch_col_count = patch_col_count;
    global_info.coding_type = coding_type;
    global_info.word_count = word_count;
    global_info.dict = dict;
    global_info.dict_transp = dict_transp;
    global_info.dict_x_dict_transp = dict_x_dict_transp;
    global_info.coeff_count = coeff_count;
    global_info.coding_params = coding_params;
    global_info.nonlinear_type = nonlinear_type;
    global_info.nonlinear_modulator = nonlinear_modulator;
    global_info.polarity_split_type = polarity_split_type;
    global_info.reduce_type = reduce_type;
    global_info.reduce_spread = reduce_spread;

    global_vars.o_sample_coded_pr = (double*)mxMalloc(global_info.new_geometry * sample_count * sizeof(double));
    global_vars.o_sample_coded_ir = (size_t*)mxMalloc(global_info.new_geometry * sample_count * sizeof(size_t));
    global_vars.o_sample_coded_jc = (size_t*)mxMalloc((sample_count + 1) * sizeof(size_t));
    global_vars.o_observations_perm = (size_t*)mxMalloc(sample_count * sizeof(size_t));
    global_vars.current_sample_coded_count = 0;
    global_vars.current_sample_coded_length = 0;
    pthread_res = pthread_mutex_init(&global_vars.coeffs_queue_control,NULL);
    check_condition(pthread_res == 0,"master:SystemError","Could not create queue mutex.");

    task_info = (struct task_info*)mxMalloc(sample_count * sizeof(struct task_info));

    for (ii = 0; ii < sample_count; ii++) {
	task_info[ii].observation_id = ii;
	task_info[ii].observation = sample + ii * geometry;
    }

    /* Run workers and compute output. */

    run_workers_x(&global_info,&global_vars,sample_count,sizeof(struct task_info),task_info,(task_fn_x_t)do_task,num_workers);

    /* Build output. */

    global_vars.o_sample_coded_pr = (double*)mxRealloc(global_vars.o_sample_coded_pr,global_vars.current_sample_coded_length * sizeof(double));
    global_vars.o_sample_coded_ir = (size_t*)mxRealloc(global_vars.o_sample_coded_ir,global_vars.current_sample_coded_length * sizeof(size_t));
    global_vars.o_sample_coded_jc[sample_count] = global_vars.current_sample_coded_length;

    output[O_SAMPLE_CODED] = mxCreateSparse(global_info.new_geometry,sample_count,0,mxREAL);
    mxSetPr(output[O_SAMPLE_CODED],global_vars.o_sample_coded_pr);
    mxSetIr(output[O_SAMPLE_CODED],global_vars.o_sample_coded_ir);
    mxSetJc(output[O_SAMPLE_CODED],global_vars.o_sample_coded_jc);
    mxSetNzmax(output[O_SAMPLE_CODED],global_vars.current_sample_coded_length);
    output[O_OBSERVATIONS_PERM] = mxCreateDoubleMatrix(sample_count,1,mxREAL);
    o_observations_perm = mxGetPr(output[O_OBSERVATIONS_PERM]);
    for (ii = 0; ii < sample_count; ii++) {
	o_observations_perm[ii] = (double)global_vars.o_observations_perm[ii];
    }

    /* Free memory and destroy objects. */

    mxFree(task_info);

    pthread_res = pthread_mutex_destroy(&global_vars.coeffs_queue_control);
    check_condition(pthread_res == 0,"master:SystemError","Could not destroy queue mutex.");
}
