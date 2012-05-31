#include <string.h>
#include <math.h>

#include "mex.h"

#include "linear.h"

#include "x_defines.h"
#include "x_common.h"

enum output_decoder {
    O_WEIGHTS  = 0,
    OUTPUTS_COUNT
};

enum input_decoder {
    I_TRAIN_SAMPLE  = 0,
    I_CLASS_INFO    = 1,
    I_METHOD_CODE   = 2,
    I_REG_PARAM     = 3,
    I_NUM_THREADS   = 4,
    I_LOGGER        = 5,
    INPUTS_COUNT
};

struct task_info {
    int                      class_1;
    int                      class_2;
    mwSize                   train_sample_count;
    mwSize                   train_sample_geometry;
    const double*            train_sample;
    const double*            labels_idx;
    const struct parameter*  param;
    double*                  results_weights;
};

static void
do_task(
    struct task_info*  task_info) {
    struct problem        local_prob;
    struct feature_node*  local_prob_x_t;
    struct model*         result_model;
    const char*           check_error;
    mwSize                class1_count;
    mwSize*               class1_found_index;
    mwSize                class1_current_index;
    mwSize*               class1_non_null_counts;
    mwSize                class1_non_null_full_count;
    mwSize*               class1_current_feature_counts;
    mwSize                class2_count;
    mwSize*               class2_found_index;
    mwSize                class2_current_index;
    mwSize*               class2_non_null_counts;
    mwSize                class2_non_null_full_count;
    mwSize*               class2_current_feature_counts;
    mwSize                ii;
    mwSize                jj;
    mwSize                idx_base;

    /* Determine number of instances of each class. */

    class1_count = 0;
    class2_count = 0;

    for (ii = 0; ii < task_info->train_sample_count; ii++) {
	if (task_info->labels_idx[ii] == task_info->class_1) {
	    class1_count = class1_count + 1;
	}

	if (task_info->labels_idx[ii] == task_info->class_2) {
	    class2_count = class2_count + 1;
	}
    }

    /* Determine which instances belong to each class. */

    class1_found_index = (mwSize*)calloc(class1_count,sizeof(mwSize));
    class1_current_index = 0;
    class2_found_index = (mwSize*)calloc(class2_count,sizeof(mwSize));
    class2_current_index = 0;

    for (ii = 0; ii < task_info->train_sample_count; ii++) {
	if (task_info->labels_idx[ii] == task_info->class_1) {
	    class1_found_index[class1_current_index] = ii;
	    class1_current_index = class1_current_index + 1;
	}

	if (task_info->labels_idx[ii] == task_info->class_2) {
	    class2_found_index[class2_current_index] = ii;
	    class2_current_index = class2_current_index + 1;
	}
    }

    /* Determine which features from each class instance are non-null. */

    class1_non_null_counts = (mwSize*)calloc(class1_count,sizeof(mwSize));
    memset(class1_non_null_counts,0,class1_count * sizeof(mwSize));
    class1_non_null_full_count = 0;
    class2_non_null_counts = (mwSize*)calloc(class2_count,sizeof(mwSize));
    memset(class2_non_null_counts,0,class2_count * sizeof(mwSize));
    class2_non_null_full_count = 0;

    for (ii = 0; ii < class1_count; ii++) {
	idx_base = class1_found_index[ii] * task_info->train_sample_geometry;

	for (jj = 0; jj < task_info->train_sample_geometry; jj++) {
	    if (task_info->train_sample[idx_base + jj] != 0) {
		class1_non_null_counts[ii] = class1_non_null_counts[ii] + 1;
		class1_non_null_full_count = class1_non_null_full_count + 1;
	    }
	}
    }

    for (ii = 0; ii < class2_count; ii++) {
	idx_base = class2_found_index[ii] * task_info->train_sample_geometry;

	for (jj = 0; jj < task_info->train_sample_geometry; jj++) {
	    if (task_info->train_sample[idx_base + jj] != 0) {
		class2_non_null_counts[ii] = class2_non_null_counts[ii] + 1;
		class2_non_null_full_count = class2_non_null_full_count + 1;
	    }
	}
    }

    /* Build "prob" problem structure. */

    local_prob.l = (int)class1_count + (int)class2_count;
    local_prob.n = (int)task_info->train_sample_geometry + 1;
    local_prob.y = (double*)calloc(class1_count + class2_count,sizeof(double));
    local_prob.x = (struct feature_node**)calloc(class1_count + class2_count,sizeof(struct feature_node*));
    local_prob.bias = 1;
    local_prob_x_t = (struct feature_node*)calloc(class1_non_null_full_count + 2 * class1_count +
						  class2_non_null_full_count + 2 * class2_count,sizeof(struct feature_node));

    local_prob.y[0] = +1;
    local_prob.x[0] = &local_prob_x_t[0];

    for (ii = 1; ii < class1_count; ii++) {
	local_prob.y[ii] = +1;
	local_prob.x[ii] = local_prob.x[ii - 1] + class1_non_null_counts[ii - 1] + 2;
    }

    local_prob.y[class1_count] = -1;
    local_prob.x[class1_count] = local_prob.x[class1_count - 1] + class1_non_null_counts[class1_count - 1] + 2;

    for (ii = 1; ii < class2_count; ii++) {
	local_prob.y[class1_count + ii] = -1;
	local_prob.x[class1_count + ii] = local_prob.x[class1_count + ii - 1] + class2_non_null_counts[ii - 1] + 2;
    }

    class1_current_feature_counts = (mwSize*)calloc(class1_count,sizeof(mwSize));
    memset(class1_current_feature_counts,0,class1_count * sizeof(mwSize));
    class2_current_feature_counts = (mwSize*)calloc(class2_count,sizeof(mwSize));
    memset(class2_current_feature_counts,0,class2_count * sizeof(mwSize));

    for (ii = 0; ii < class1_count; ii++) {
	idx_base = class1_found_index[ii] * task_info->train_sample_geometry;

	for (jj = 0; jj < task_info->train_sample_geometry; jj++) {
	    if (task_info->train_sample[idx_base + jj] != 0) {
		local_prob.x[ii][class1_current_feature_counts[ii]].index = (int)jj + 1;
		local_prob.x[ii][class1_current_feature_counts[ii]].value = task_info->train_sample[idx_base + jj];
		class1_current_feature_counts[ii] = class1_current_feature_counts[ii] + 1;
	    }
	}
    }

    for (ii = 0; ii < class2_count; ii++) {
	idx_base = class2_found_index[ii] * task_info->train_sample_geometry;

	for (jj = 0; jj < task_info->train_sample_geometry; jj++) {
	    if (task_info->train_sample[idx_base + jj] != 0) {
		local_prob.x[class1_count + ii][class2_current_feature_counts[ii]].index = (int)jj + 1;
		local_prob.x[class1_count + ii][class2_current_feature_counts[ii]].value = task_info->train_sample[idx_base + jj];
		class2_current_feature_counts[ii] = class2_current_feature_counts[ii] + 1;
	    }
	}
    }

    for (ii = 0; ii < class1_count; ii++) {
	local_prob.x[ii][class1_current_feature_counts[ii]].index = (int)task_info->train_sample_geometry + 1;
	local_prob.x[ii][class1_current_feature_counts[ii]].value = 1;
    }

    for (ii = 0; ii < class1_count; ii++) {
	local_prob.x[ii][class1_current_feature_counts[ii] + 1].index = -1;
	local_prob.x[ii][class1_current_feature_counts[ii] + 1].value = 0;
    }

    for (ii = 0; ii < class2_count; ii++) {
	local_prob.x[class1_count + ii][class2_current_feature_counts[ii]].index = (int)task_info->train_sample_geometry + 1;
	local_prob.x[class1_count + ii][class2_current_feature_counts[ii]].value = 1;
    }

    for (ii = 0; ii < class2_count; ii++) {
	local_prob.x[class1_count + ii][class2_current_feature_counts[ii] + 1].index = -1;
	local_prob.x[class1_count + ii][class2_current_feature_counts[ii] + 1].value = 0;
    }

    /* Call "check_parameter" to validate our problem and parameters structures. */

    check_error = check_parameter(&local_prob,task_info->param);
    check_condition(check_error == NULL,"master:NoConvergence",check_error);

    /* Train with local problem. */

    result_model = train(&local_prob,task_info->param);

    memcpy(task_info->results_weights,result_model->w,(task_info->train_sample_geometry + 1) * sizeof(double));

    /* Free memory. */

    free_model_content(result_model);

    free(class2_current_feature_counts);
    free(class1_current_feature_counts);
    free(local_prob_x_t);
    free(local_prob.x);
    free(local_prob.y);
    free(class2_non_null_counts);
    free(class1_non_null_counts);
    free(class2_found_index);
    free(class1_found_index);
}

void
mexFunction(
    int             output_count,
    mxArray*        output[],
    int             input_count,
    const mxArray*  input[]) {
    mwSize                train_sample_count;
    mwSize                train_sample_geometry;
    const double*         train_sample;
    int                   classes_count;
    const double*         labels_idx;
    int                   method_code;
    double                reg_param;
    int                   num_threads;
    mxArray*              local_logger;
    int                   classifiers_count;
    struct parameter      param;
    struct task_info*     task_info;
    double*               task_info_results_weights_t;
    int                   classifier_index;
    int                   ii_int;
    int                   jj_int;

    /* Validate "input" and "output" parameters. */

    check_condition(output_count == OUTPUTS_COUNT,
		    "master:InvalidMEXCall","Invalid number of outputs.");
    check_condition(input_count == INPUTS_COUNT,
		    "master:InvalidMEXCall","Invalid number of inputs.");
    check_condition(mxGetNumberOfDimensions(input[I_TRAIN_SAMPLE]) == 2,
		    "master:InvalidMEXCall","Parameter \"train_sample\" is not a tc.dataset_record.");
    check_condition(mxGetM(input[I_TRAIN_SAMPLE]) >= 1,
		    "master:InvalidMEXCall","Parameter \"train_sample\" is not a tc.dataset_record.");
    check_condition(mxGetN(input[I_TRAIN_SAMPLE]) >= 1,
		    "master:InvalidMEXCall","Parameter \"train_sample\" is not a tc.dataset_record.");
    check_condition(mxIsDouble(input[I_TRAIN_SAMPLE]),
		    "master:InvalidMEXCall","Parameter \"train_sample\" is not a tc.dataset_record.");
    check_condition(mxGetNumberOfDimensions(input[I_CLASS_INFO]) == 2,
		    "master:InvalidMEXCall","Parameter \"class_info\" is not a tc.scalar.");
    check_condition(mxGetM(input[I_CLASS_INFO]) == 1,
		    "master:InvalidMEXCall","Parameter \"class_info\" is not a tc.scalar.");
    check_condition(mxGetN(input[I_CLASS_INFO]) == 1,
		    "master:InvalidMEXCall","Parameter \"class_info\" is not a tc.scalar.");
    check_condition(strcmp(mxGetClassName(input[I_CLASS_INFO]),"classification_info") == 0,
		    "master:InvalidMEXCall","Parameter \"class_info\" is not a tc.classification_info.");
    check_condition(mxGetNumberOfDimensions(input[I_METHOD_CODE]) == 2,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a tc.scalar.");
    check_condition(mxGetM(input[I_METHOD_CODE]) == 1,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a tc.scalar.");
    check_condition(mxGetN(input[I_METHOD_CODE]) == 1,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a tc.scalar.");
    check_condition(mxIsDouble(input[I_METHOD_CODE]),
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a tc.natural.");
    check_condition(fabs(mxGetScalar(input[I_METHOD_CODE]) - floor(mxGetScalar(input[I_METHOD_CODE]))) == 0,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a tc.natural.");
    check_condition(mxGetScalar(input[I_METHOD_CODE]) >= 0,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a tc.natural.");
    check_condition(mxGetScalar(input[I_METHOD_CODE]) < INT_MAX,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a tc.natural.");
    check_condition((mxGetScalar(input[I_METHOD_CODE]) == L2R_LR) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L2R_L2LOSS_SVC_DUAL) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L2R_L2LOSS_SVC) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L2R_L1LOSS_SVC_DUAL) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L1R_L2LOSS_SVC) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L1R_LR) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L2R_LR_DUAL),
		    "master:InvalidMEXCall","Parameter \"method_code\" has an invalid value.");
    check_condition(mxGetNumberOfDimensions(input[I_REG_PARAM]) == 2,
		    "master:InvalidMEXCall","Parameter \"reg_param\" is not a tc.scalar.");
    check_condition(mxGetM(input[I_REG_PARAM]) == 1,
		    "master:InvalidMEXCall","Parameter \"reg_param\" is not a tc.scalar.");
    check_condition(mxGetN(input[I_REG_PARAM]) == 1,
		    "master:InvalidMEXCall","Parameter \"reg_param\" is not a tc.scalar.");
    check_condition(mxIsDouble(input[I_REG_PARAM]),
		    "master:InvalidMEXCall","Parameter \"reg_param\" is not a tc.number.");
    check_condition(mxGetScalar(input[I_REG_PARAM]) > 0,
		    "master:InvalidMEXCall","Parameter \"reg_param\" is not strictly positive.");
    check_condition(mxGetNumberOfDimensions(input[I_NUM_THREADS]) == 2,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a tc.scalar.");
    check_condition(mxGetM(input[I_NUM_THREADS]) == 1,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a tc.scalar.");
    check_condition(mxGetN(input[I_NUM_THREADS]) == 1,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a tc.scalar.");
    check_condition(mxIsDouble(input[I_NUM_THREADS]),
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a tc.natural.");
    check_condition(fabs(mxGetScalar(input[I_NUM_THREADS]) - floor(mxGetScalar(input[I_NUM_THREADS]))) == 0,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a tc.natural.");
    check_condition(mxGetScalar(input[I_NUM_THREADS]) >= 0,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a tc.natural.");
    check_condition(mxGetScalar(input[I_NUM_THREADS]) < INT_MAX,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a tc.natural.");
    check_condition(input[I_NUM_THREADS] > 0,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not strictly positive.");
    check_condition(mxGetNumberOfDimensions(input[I_LOGGER]) == 2,
		    "master:InvalidMEXCall","Parameter \"logger\" is not a tc.scalar.");
    check_condition(mxGetM(input[I_LOGGER]) == 1,
		    "master:InvalidMEXCall","Parameter \"logger\" is not a tc.scalar.");
    check_condition(mxGetN(input[I_LOGGER]) == 1,
		    "master:InvalidMEXCall","Parameter \"logger\" is not a tc.scalar.");
    check_condition(strcmp(mxGetClassName(input[I_LOGGER]),"logging.logger") == 0,
		    "master:InvalidMEXCall","Parameter \"logger\" is not a tc.logging_logger.");
    check_condition(mxGetScalar(mxGetProperty(input[I_LOGGER],0,"active")) == 1,
    		    "master:InvalidMEXCall","Parameter \"logger\" is not active.");
    check_condition(mxGetN(input[I_TRAIN_SAMPLE]) == mxGetN(mxGetProperty(input[I_CLASS_INFO],0,"labels_idx")),
		    "master:InvalidMEXCall","Different number of sample instances and label indices.");
    check_condition(mxGetN(input[I_TRAIN_SAMPLE]) < INT_MAX,
		    "master:InvalidMEXCall","Too many sample instances for \"liblinear\".");
    check_condition(mxGetM(input[I_TRAIN_SAMPLE]) < INT_MAX - 1,
		    "master:InvalidMEXCall","Too many features for \"liblinear\".");
    check_condition(mxGetScalar(mxGetProperty(input[I_CLASS_INFO],0,"labels_count")) < INT_MAX,
		    "master:InvalidMEXCall","Too many classes for \"liblinear\".");

    /* For proper output in MATLAB we set this to a correct-type wrapper around "mexPrintf". */

    set_print_string_function(printf_wrapper);

    /* Extract relevant information from all inputs. */

    train_sample_count = mxGetN(input[I_TRAIN_SAMPLE]);
    train_sample_geometry = mxGetM(input[I_TRAIN_SAMPLE]);
    train_sample = mxGetPr(input[I_TRAIN_SAMPLE]);
    classes_count = (int)mxGetScalar(mxGetProperty(input[I_CLASS_INFO],0,"labels_count"));
    labels_idx = mxGetPr(mxGetProperty(input[I_CLASS_INFO],0,"labels_idx"));
    method_code = (int)mxGetScalar(input[I_METHOD_CODE]);
    reg_param = mxGetScalar(input[I_REG_PARAM]); 
    num_threads = (int)mxGetScalar(input[I_NUM_THREADS]);
    local_logger = mxDuplicateArray(input[I_LOGGER]);

    classifiers_count = classes_count * (classes_count - 1) / 2;

    logger_beg_node(local_logger,"Parallel training via \"liblinear\" in One-vs-One fashion");

    logger_beg_node(local_logger,"Passed configuration");

    logger_message(local_logger,"Train sample count: %d",train_sample_count);
    logger_message(local_logger,"Train sample geometry: %d",train_sample_geometry);
    logger_message(local_logger,"Classes count: %d",classes_count);
    logger_message(local_logger,"Method: %s",METHOD_CODE_TO_STRING[method_code]);
    logger_message(local_logger,"Regularization parameter: %.3f",reg_param);
    logger_message(local_logger,"Number of worker threads: %d",num_threads);
    logger_message(local_logger,"Classifiers count: %d",classifiers_count);

    logger_end_node(local_logger);

    /* Build "param" parameter structure. */

    logger_message(local_logger,"Building parameters info structure.");

    param.solver_type = method_code;
    param.eps = EPS_DEFAULT[method_code];
    param.C = reg_param;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.p = 0;

    logger_beg_node(local_logger,"Parameters structure [Sanity Check]");

    logger_message(local_logger,"Solver type: %s",METHOD_CODE_TO_STRING[param.solver_type]);
    logger_message(local_logger,"Epsilon: %f",param.eps);
    logger_message(local_logger,"Regularization param: %.3f",param.C);
    logger_message(local_logger,"Number of weight biases: %d",param.nr_weight);
    logger_message(local_logger,"SVR p: %.3f",param.p);

    logger_end_node(local_logger);

    /* Build thread pool. */

    logger_message(local_logger,"Building worker task allocation.");

    task_info = (struct task_info*)mxCalloc(classifiers_count,sizeof(struct task_info));
    task_info_results_weights_t = (double*)mxCalloc(classifiers_count * (train_sample_geometry + 1),sizeof(double));

    classifier_index = 0;

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	for (jj_int = ii_int + 1; jj_int < classes_count; jj_int++) {
	    task_info[classifier_index].class_1 = ii_int + 1;
	    task_info[classifier_index].class_2 = jj_int + 1;
	    task_info[classifier_index].train_sample_count = train_sample_count;
	    task_info[classifier_index].train_sample_geometry = train_sample_geometry;
	    task_info[classifier_index].train_sample = train_sample;
	    task_info[classifier_index].labels_idx = labels_idx;
	    task_info[classifier_index].param = &param;
	    task_info[classifier_index].results_weights = task_info_results_weights_t + classifier_index * (train_sample_geometry + 1);
	    classifier_index = classifier_index + 1;
	}
    }

    /* Start worker threads. */

    logger_message(local_logger,"Starting parallel training of classifiers.");

    run_workers(num_threads,(task_fn_t)do_task,classifiers_count,task_info,sizeof(struct task_info));

    logger_end_node(local_logger);

    /* Build "output". */

    output[O_WEIGHTS] = mxCreateDoubleMatrix(0,0,mxREAL);
    mxSetPr(output[O_WEIGHTS],task_info_results_weights_t);
    mxSetM(output[O_WEIGHTS],train_sample_geometry + 1);
    mxSetN(output[O_WEIGHTS],classifiers_count);

    /* Free memory. */

    mxFree(task_info);
    mxDestroyArray(local_logger);
}
