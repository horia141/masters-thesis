#include "mex.h"

#include "x_mex_interface.h"
#include "base_defines.h"
#include "task_control.h"
#include "coding_methods.h"
#include "latools.h"

enum output_decoder {
    O_COEFFS  = 0,
    OUTPUTS_COUNT
};

enum input_decoder {
    I_DICT               = 0,
    I_DICT_TRANSP        = 1,
    I_DICT_X_DICT_TRANSP = 2,
    I_PARAMS             = 3,
    I_COEFF_COUNT        = 4,
    I_SAMPLE             = 5,
    I_NUM_WORKERS        = 6,
    INPUTS_COUNT
};

struct global_info {
    size_t         geometry;
    size_t         word_count;
    const double*  dict;
    const double*  dict_transp;
    const double*  dict_x_dict_transp;
    size_t         coeff_count;
};

struct task_info {
    double*        o_coeffs_pr;
    size_t*        o_coeffs_ir;
    const double*  observation;
};

static void
do_task(
    size_t                     id,
    const struct global_info*  global_info,
    void*                      global_vars,
    size_t                     task_info_count,
    struct task_info*          task_info) {
    char*   coding_tmps;
    size_t  ii;

    coding_tmps = (char*)malloc(global_info->word_count * sizeof(double) + global_info->word_count * sizeof(size_t));

    for (ii = 0; ii < task_info_count; ii++) {
        correlation(task_info[ii].o_coeffs_pr,task_info[ii].o_coeffs_ir,
		    global_info->geometry,global_info->word_count,global_info->dict,global_info->dict_transp,global_info->dict_x_dict_transp,
		    global_info->coeff_count,NULL,task_info[ii].observation,coding_tmps);
	sort_by_idxs(task_info[ii].o_coeffs_pr,task_info[ii].o_coeffs_ir,global_info->coeff_count);
    }

    free(coding_tmps);
}

void
mexFunction(
    int             output_count,
    mxArray*        output[],
    int             input_count,
    const mxArray*  input[]) {
    size_t               geometry;
    size_t               word_count;
    const double*        dict;
    const double*        dict_transp;
    const double*        dict_x_dict_transp;
    size_t               coeff_count;
    size_t               sample_count;
    const double*        sample;
    size_t               num_workers;
    mxArray*             o_coeffs;
    double*              o_coeffs_pr;
    mwIndex*             o_coeffs_ir;
    mwIndex*             o_coeffs_jc;
    struct global_info   global_info;
    struct task_info*    task_info;
    size_t               ii;

    /* Extract relevant information from all inputs. */

    geometry = mxGetM(input[I_SAMPLE]);
    word_count = mxGetM(input[I_DICT]);
    dict = mxGetPr(input[I_DICT]);
    dict_transp = mxGetPr(input[I_DICT_TRANSP]);
    dict_x_dict_transp = mxGetPr(input[I_DICT_X_DICT_TRANSP]);
    coeff_count = (size_t)mxGetScalar(input[I_COEFF_COUNT]);
    sample_count = mxGetN(input[I_SAMPLE]);
    sample = mxGetPr(input[I_SAMPLE]);
    num_workers = (size_t)mxGetScalar(input[I_NUM_WORKERS]);

    /* Build output structures. */

    o_coeffs = mxCreateSparse(word_count,sample_count,coeff_count * sample_count,mxREAL);
    o_coeffs_pr = mxGetPr(o_coeffs);
    o_coeffs_ir = mxGetIr(o_coeffs);
    o_coeffs_jc = mxGetJc(o_coeffs);

    for (ii = 0; ii <= sample_count; ii++) {
        o_coeffs_jc[ii] = ii * coeff_count;
    }

    /* Build task distribution information. */

    global_info.geometry = geometry;
    global_info.word_count = word_count;
    global_info.dict = dict;
    global_info.dict_transp = dict_transp;
    global_info.dict_x_dict_transp = dict_x_dict_transp;
    global_info.coeff_count = coeff_count;

    task_info = (struct task_info*)mxMalloc(sample_count * sizeof(struct task_info));

    for (ii = 0; ii < sample_count; ii++) {
        task_info[ii].o_coeffs_pr = o_coeffs_pr + ii * coeff_count;
        task_info[ii].o_coeffs_ir = o_coeffs_ir + ii * coeff_count;
        task_info[ii].observation = sample + ii * geometry;
    }

    /* Run workers and compute output. */

    run_workers_x(&global_info,NULL,sample_count,sizeof(struct task_info),task_info,(task_fn_x_t)do_task,num_workers);

    /* Build "output". */

    output[O_COEFFS] = o_coeffs;

    /* Free memory. */

    mxFree(task_info);
}
