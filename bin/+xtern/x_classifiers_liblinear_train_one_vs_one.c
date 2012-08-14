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
    const double*            train_sample_pr;
    const size_t*            train_sample_ir;
    const size_t*            train_sample_jc;
    const double*            labels_idx;
    const struct parameter*  param;
};

struct task_info {
    double*  o_weights;
    int      class_1;
    int      class_2;
};

static void
do_task(
    size_t                     id,
    const struct global_info*  global_info,
    struct global_vars*        global_vars,
    size_t                     task_info_count,
    struct task_info*          task_info) {
    struct problem        local_prob;
    struct feature_node*  prob_x_t;
    size_t                current_observation;
    struct feature_node*  prob_x_t_curr;
    const double*         train_sample_pr_curr;
    const size_t*         train_sample_ir_curr;
    size_t                observation_count;
    struct model*         result_model;
    const char*           check_error;
    size_t                class1_count;
    size_t                class1_space_count;
    size_t                class2_count;
    size_t                class2_space_count;
    size_t                ii;
    size_t                jj;
    size_t                kk;

    for (ii = 0; ii < task_info_count; ii++) {
	/* Determine number of instances of each class. */

	class1_count = 0;
	class1_space_count = 0;
	class2_count = 0;
	class2_space_count = 0;

	for (jj = 0; jj < global_info->train_sample_count; jj++) {
	    if (global_info->labels_idx[jj] == task_info[ii].class_1) {
		class1_count = class1_count + 1;
		class1_space_count = class1_space_count + global_info->train_sample_jc[jj + 1] - global_info->train_sample_jc[jj];
	    } else if (global_info->labels_idx[jj] == task_info[ii].class_2) {
		class2_count = class2_count + 1;
		class2_space_count = class2_space_count + global_info->train_sample_jc[jj + 1] - global_info->train_sample_jc[jj];
	    }
	}

	local_prob.l = (int)class1_count + (int)class2_count;
	local_prob.n = (int)global_info->geometry + 1;
	local_prob.y = (double*)malloc((class1_count + class2_count) * sizeof(double));
	local_prob.x = (struct feature_node**)malloc((class1_count + class2_count) * sizeof(struct feature_node*));
	local_prob.bias = 1;
	prob_x_t = (struct feature_node*)malloc((class1_space_count + 2 * class1_count + class2_space_count + 2 * class2_count) * sizeof(struct feature_node));

	current_observation = 0;
	prob_x_t_curr = prob_x_t;
	train_sample_pr_curr = global_info->train_sample_pr;
	train_sample_ir_curr = global_info->train_sample_ir;

	for (jj = 0; jj < global_info->train_sample_count; jj++) {
	    observation_count = global_info->train_sample_jc[jj + 1] - global_info->train_sample_jc[jj];

	    if (global_info->labels_idx[jj] == task_info[ii].class_1) {
		local_prob.x[current_observation] = prob_x_t_curr;

		for (kk = 0; kk < observation_count; kk++) {
		    prob_x_t_curr[kk].index = (int)train_sample_ir_curr[kk] + 1;
		    prob_x_t_curr[kk].value = train_sample_pr_curr[kk];
		}

		prob_x_t_curr[observation_count + 0].index = global_info->geometry + 1;
		prob_x_t_curr[observation_count + 0].value = 1;
		prob_x_t_curr[observation_count + 1].index = -1;
		prob_x_t_curr[observation_count + 1].value = 0;

		local_prob.y[current_observation] = +1;

		current_observation = current_observation + 1;
		prob_x_t_curr = prob_x_t_curr + observation_count + 2;
	    }

	    train_sample_pr_curr = train_sample_pr_curr + observation_count;
	    train_sample_ir_curr = train_sample_ir_curr + observation_count;
	}

	train_sample_pr_curr = global_info->train_sample_pr;
	train_sample_ir_curr = global_info->train_sample_ir;

	for (jj = 0; jj < global_info->train_sample_count; jj++) {
	    observation_count = global_info->train_sample_jc[jj + 1] - global_info->train_sample_jc[jj];

	    if (global_info->labels_idx[jj] == task_info[ii].class_2) {
		local_prob.x[current_observation] = prob_x_t_curr;

		for (kk = 0; kk < observation_count; kk++) {
		    prob_x_t_curr[kk].index = (int)train_sample_ir_curr[kk] + 1;
		    prob_x_t_curr[kk].value = train_sample_pr_curr[kk];
		}

		prob_x_t_curr[observation_count + 0].index = global_info->geometry + 1;
		prob_x_t_curr[observation_count + 0].value = 1;
		prob_x_t_curr[observation_count + 1].index = -1;
		prob_x_t_curr[observation_count + 1].value = 0;

		local_prob.y[current_observation] = -1;

		current_observation = current_observation + 1;
		prob_x_t_curr = prob_x_t_curr + observation_count + 2;
	    }

	    train_sample_pr_curr = train_sample_pr_curr + observation_count;
	    train_sample_ir_curr = train_sample_ir_curr + observation_count;
	}

	/* Call "check_parameter" to validate our problem and parameters structures. */

	check_error = check_parameter(&local_prob,global_info->param);
	check_condition(check_error == NULL,"master:NoConvergence",check_error);

	/* Train with local problem. */

	result_model = train(&local_prob,global_info->param);

	memcpy(task_info[ii].o_weights,result_model->w,(global_info->geometry + 1) * sizeof(double));

	/* Free memory. */

	free_model_content(result_model);
	free(prob_x_t);
	free(local_prob.x);
	free(local_prob.y);
    }
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
    const double*         labels_idx;
    int                   method_code;
    double                reg_param;
    int                   num_workers;
    int                   classifiers_count;
    struct parameter      param;
    struct global_info    global_info;
    struct task_info*     task_info;
    double*               task_info_weights_t;
    int                   classifier_index;
    int                   ii;
    int                   jj;

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

    classifiers_count = classes_count * (classes_count - 1) / 2;

    /* Build parameter structure. */

    param.solver_type = method_code;
    param.eps = EPS_DEFAULT[method_code];
    param.C = reg_param;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.p = 0;

    /* Build task distribution information. */

    global_info.geometry = geometry;
    global_info.train_sample_count = train_sample_count;
    global_info.train_sample_pr = train_sample_pr;
    global_info.train_sample_ir = train_sample_ir;
    global_info.train_sample_jc = train_sample_jc;
    global_info.labels_idx = labels_idx;
    global_info.param = &param;

    task_info = (struct task_info*)mxMalloc(classifiers_count * sizeof(struct task_info));
    task_info_weights_t = (double*)mxMalloc(classifiers_count * (geometry + 1) * sizeof(double));

    classifier_index = 0;

    for (ii = 0; ii < classes_count; ii++) {
	for (jj = ii + 1; jj < classes_count; jj++) {
	    task_info[classifier_index].o_weights = task_info_weights_t + classifier_index * (geometry + 1);
	    task_info[classifier_index].class_1 = ii + 1;
	    task_info[classifier_index].class_2 = jj + 1;

	    classifier_index = classifier_index + 1;
	}
    }

    /* Start worker workers. */

    run_workers_x(&global_info,NULL,classifiers_count,sizeof(struct task_info),task_info,(task_fn_x_t)do_task,num_workers);

    /* Build "output". */

    output[O_WEIGHTS] = mxCreateDoubleMatrix(0,0,mxREAL);
    mxSetPr(output[O_WEIGHTS],task_info_weights_t);
    mxSetM(output[O_WEIGHTS],geometry + 1);
    mxSetN(output[O_WEIGHTS],classifiers_count);

    /* Free memory. */

    mxFree(task_info);
}
