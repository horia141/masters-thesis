#include <string.h>

#include "mex.h"

#include "liblinear/linear.h"

#include "x_mex_interface.h"
#include "x_classifiers_liblinear_defines.h"
#include "task_control.h"

enum output_decoder {
    O_WEIGHTS  = 0,
    OUTPUTS_COUNT
};

enum input_decoder {
    I_TRAIN_SAMPLE  = 0,
    I_CLASS_INFO    = 1,
    I_METHOD_CODE   = 2,
    I_REG_PARAM     = 3,
    I_NUM_WORKERS   = 4,
    INPUTS_COUNT
};

struct global_info {
    size_t                   geometry;
    size_t                   train_sample_count;
    const struct problem*    prob;
    const struct parameter*  param;
};

struct task_info {
    double*  o_weights;
    int      class_all;
};

static void
do_task(
    size_t                     id,
    const struct global_info*  global_info,
    struct global_vars*        global_vars,
    size_t                     task_info_count,
    struct task_info*          task_info) {
    struct problem        local_prob;
    struct feature_node*  temp_features;
    double                temp_label;
    struct model*         result_model;
    size_t                ii;
    size_t                jj;

    local_prob.l = (int)global_info->train_sample_count;
    local_prob.n = (int)global_info->geometry + 1;
    local_prob.y = (double*)malloc(global_info->train_sample_count * sizeof(double));
    local_prob.x = (struct feature_node**)malloc(global_info->train_sample_count * sizeof(struct feature_node*));
    local_prob.bias = global_info->prob->bias;

    for (ii = 0; ii < task_info_count; ii++) {
	memcpy(local_prob.x,global_info->prob->x,global_info->train_sample_count * sizeof(struct feature_node*));

	/* Separate instances into "class" and "all" groups. */

	for (jj = 0; jj < global_info->train_sample_count; jj++) {
	    if (global_info->prob->y[jj] == task_info[ii].class_all) {
		local_prob.y[jj] = +1;
	    } else {
		local_prob.y[jj] = -1;
	    }
	}

	/* Make sure the first instance is of the "+1" class. This is needed so that the surface normal
	   points "towards" the "class" instances and we get sane prediction in "x_do_classify". */

	for (jj = 0; jj < global_info->train_sample_count; jj++) {
	    if (global_info->prob->y[jj] == task_info[ii].class_all) {
		temp_features = local_prob.x[0];
		local_prob.x[0] = local_prob.x[jj];
		local_prob.x[jj] = temp_features;

		temp_label = local_prob.y[0];
		local_prob.y[0] = local_prob.y[jj];
		local_prob.y[jj] = temp_label;

		break;
	    }
	}

	/* Train the binary classifier and copy the resulting surface normal ("weights") into the results
	   buffer. */

	result_model = train(&local_prob,global_info->param);

	memcpy(task_info[ii].o_weights,result_model->w,(global_info->geometry + 1) * sizeof(double));

	free_model_content(result_model);
    }

    free(local_prob.x);
    free(local_prob.y);
}

void
mexFunction(
    int             output_count,
    mxArray*        output[],
    int             input_count,
    const mxArray*  input[]) {
    size_t                geometry;
    size_t                train_sample_count;
    const double*         train_sample_pr;
    const size_t*         train_sample_ir;
    const size_t*         train_sample_jc;
    int                   classes_count;
    double*               labels_idx;
    int                   method_code;
    double                reg_param;
    int                   num_workers;
    int                   classifiers_count;
    const char*           check_error;
    struct problem        prob;
    struct feature_node*  prob_x_t;
    struct feature_node*  prob_x_t_curr;
    const double*         train_sample_pr_curr;
    const size_t*         train_sample_ir_curr;
    size_t                observation_count; 
    struct parameter      param;
    struct global_info    global_info;
    struct task_info*     task_info;
    double*               task_info_weights_t;
    size_t                ii;
    size_t                jj;
    int                   kk;

    /* For proper output in MATLAB we set this to a correct-type wrapper around "mexPrintf". */

    set_print_string_function(printf_wrapper);

    /* Extract relevant information from all inputs. */

    geometry = mxGetM(input[I_TRAIN_SAMPLE]);
    train_sample_count = mxGetN(input[I_TRAIN_SAMPLE]);
    train_sample_pr = mxGetPr(input[I_TRAIN_SAMPLE]);
    train_sample_ir = mxGetIr(input[I_TRAIN_SAMPLE]);
    train_sample_jc = mxGetJc(input[I_TRAIN_SAMPLE]);
    classes_count = (int)mxGetScalar(mxGetProperty(input[I_CLASS_INFO],0,"labels_count"));
    labels_idx = mxGetPr(mxGetProperty(input[I_CLASS_INFO],0,"labels_idx"));
    method_code = (int)mxGetScalar(input[I_METHOD_CODE]);
    reg_param = mxGetScalar(input[I_REG_PARAM]); 
    num_workers = (int)mxGetScalar(input[I_NUM_WORKERS]);

    classifiers_count = classes_count;

    /* Build problem and parameter structures. */

    prob.l = (int)train_sample_count;
    prob.n = (int)geometry + 1;
    prob.y = labels_idx;
    prob.x = (struct feature_node**)mxMalloc(train_sample_count * sizeof(struct feature_node*));
    prob.bias = 1;

    prob_x_t = (struct feature_node*)mxMalloc((train_sample_jc[train_sample_count] + 2*train_sample_count) * sizeof(struct feature_node));

    prob_x_t_curr = prob_x_t;
    train_sample_pr_curr = train_sample_pr;
    train_sample_ir_curr = train_sample_ir;

    for (ii = 0; ii < train_sample_count; ii++) {
	observation_count = train_sample_jc[ii + 1] - train_sample_jc[ii];
	prob.x[ii] = prob_x_t_curr;

	for (jj = 0; jj < observation_count; jj++) {
	    prob_x_t_curr[jj].index = (int)train_sample_ir_curr[jj] + 1;
	    prob_x_t_curr[jj].value = train_sample_pr_curr[jj];
	}

	prob_x_t_curr[observation_count + 0].index = geometry + 1;
	prob_x_t_curr[observation_count + 0].value = 1;
	prob_x_t_curr[observation_count + 1].index = -1;
	prob_x_t_curr[observation_count + 1].value = 0;

	prob_x_t_curr = prob_x_t_curr + observation_count + 2;
	train_sample_pr_curr = train_sample_pr_curr + observation_count;
	train_sample_ir_curr = train_sample_ir_curr + observation_count;
    }

    param.solver_type = method_code;
    param.eps = EPS_DEFAULT[method_code];
    param.C = reg_param;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.p = 0;

    /* Call "check_parameter" to validate our problem and parameters structures. */

    check_error = check_parameter(&prob,&param);
    check_condition(check_error == NULL,"master:NoConvergence",check_error);

    /* Build task distribution information. */

    global_info.geometry = geometry;
    global_info.train_sample_count = train_sample_count;
    global_info.prob = &prob;
    global_info.param = &param;

    task_info = (struct task_info*)mxMalloc(classifiers_count * sizeof(struct task_info));
    task_info_weights_t = (double*)mxMalloc(classifiers_count * (geometry + 1) * sizeof(double));

    for (kk = 0; kk < classifiers_count; kk++) {
	task_info[kk].o_weights = task_info_weights_t + kk * (geometry + 1);
	task_info[kk].class_all = kk + 1;
    }

    /* Run workers and compute output. */

    run_workers_x(&global_info,NULL,classifiers_count,sizeof(struct task_info),task_info,(task_fn_x_t)do_task,num_workers);

    /* Build "output". */

    output[O_WEIGHTS] = mxCreateDoubleMatrix(0,0,mxREAL);
    mxSetPr(output[O_WEIGHTS],task_info_weights_t);
    mxSetM(output[O_WEIGHTS],geometry + 1);
    mxSetN(output[O_WEIGHTS],classifiers_count);

    /* Free memory. */

    mxFree(task_info);
    mxFree(prob_x_t);
    mxFree(prob.x);
}
