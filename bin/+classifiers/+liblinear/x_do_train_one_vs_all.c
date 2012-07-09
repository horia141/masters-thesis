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
    int                      class_all;
    mwSize                   train_sample_count;
    mwSize                   train_sample_geometry;
    const struct problem*    prob;
    const struct parameter*  param;
    double*                  results_weights;
};

static void
do_task(
    struct task_info*  task_info) {
    struct problem        local_prob;
    struct feature_node*  temp_features;
    struct model*         result_model;
    double                temp_label;
    mwSize                ii;

    /* Build local copy of the problem structure. The labels information will be changed
       though, to reflect the One-Vs-All type of multiclassification that we're doing. */

    local_prob.l = (int)task_info->train_sample_count;
    local_prob.n = (int)task_info->train_sample_geometry + 1;
    local_prob.y = (double*)calloc(task_info->train_sample_count,sizeof(double));
    local_prob.x = (struct feature_node**)calloc(task_info->train_sample_count,sizeof(struct feature_node*));
    local_prob.bias = task_info->prob->bias;

    memcpy(local_prob.x,task_info->prob->x,task_info->train_sample_count * sizeof(struct feature_node*));

    /* Separate instances into "class" and "all" groups. */

    for (ii = 0; ii < task_info->train_sample_count; ii++) {
	if (task_info->prob->y[ii] == task_info->class_all) {
	    local_prob.y[ii] = +1;
	} else {
	    local_prob.y[ii] = -1;
	}
    }

    /* Make sure the first instance is of the "+1" class. This is needed so that the surface normal
       points "towards" the "class" instances and we get sane prediction in "x_do_classify". */

    for (ii = 0; ii < task_info->train_sample_count; ii++) {
	if (task_info->prob->y[ii] == task_info->class_all) {
	    temp_features = local_prob.x[0];
	    local_prob.x[0] = local_prob.x[ii];
	    local_prob.x[ii] = temp_features;

	    temp_label = local_prob.y[0];
	    local_prob.y[0] = local_prob.y[ii];
	    local_prob.y[ii] = temp_label;

	    break;
	}
    }

    /* Train the binary classifier and copy the resulting surface normal ("weights") into the results
       buffer. */

    result_model = train(&local_prob,task_info->param);

    memcpy(task_info->results_weights,result_model->w,(task_info->train_sample_geometry + 1) * sizeof(double));

    /* Free memory. */

    free_model_content(result_model);
    free(local_prob.x);
    free(local_prob.y);
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
    double*               labels_idx;
    int                   method_code;
    double                reg_param;
    int                   num_threads;
    mxArray*              local_logger;
    int                   classifiers_count;
    mwSize*               non_null_counts;
    mwSize                non_null_full_count;
    mwSize*               current_feature_counts;
    struct problem        prob;
    struct feature_node*  prob_x_t;
    struct parameter      param;
    const char*           check_error;
    struct task_info*     task_info;
    double*               task_info_results_weights_t;
    mwSize                ii;
    mwSize                jj;
    mwSize                idx_base;
    int                   ii_int;

    /* Validate "input" and "output" parameters. */

    check_condition(output_count == OUTPUTS_COUNT,
		    "master:InvalidMEXCall","Invalid number of outputs.");
    check_condition(input_count == INPUTS_COUNT,
		    "master:InvalidMEXCall","Invalid number of inputs.");
    check_condition(mxGetNumberOfDimensions(input[I_TRAIN_SAMPLE]) == 2,
		    "master:InvalidMEXCall","Parameter \"train_sample\" is not a check.dataset_record.");
    check_condition(mxGetM(input[I_TRAIN_SAMPLE]) >= 1,
		    "master:InvalidMEXCall","Parameter \"train_sample\" is not a check.dataset_record.");
    check_condition(mxGetN(input[I_TRAIN_SAMPLE]) >= 1,
		    "master:InvalidMEXCall","Parameter \"train_sample\" is not a check.dataset_record.");
    check_condition(mxIsDouble(input[I_TRAIN_SAMPLE]),
		    "master:InvalidMEXCall","Parameter \"train_sample\" is not a check.dataset_record.");
    check_condition(mxGetNumberOfDimensions(input[I_CLASS_INFO]) == 2,
		    "master:InvalidMEXCall","Parameter \"class_info\" is not a check.scalar.");
    check_condition(mxGetM(input[I_CLASS_INFO]) == 1,
		    "master:InvalidMEXCall","Parameter \"class_info\" is not a check.scalar.");
    check_condition(mxGetN(input[I_CLASS_INFO]) == 1,
		    "master:InvalidMEXCall","Parameter \"class_info\" is not a check.scalar.");
    check_condition(strcmp(mxGetClassName(input[I_CLASS_INFO]),"classifier_info") == 0,
		    "master:InvalidMEXCall","Parameter \"class_info\" is not a check.classifier_info.");
    check_condition(mxGetNumberOfDimensions(input[I_METHOD_CODE]) == 2,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a check.scalar.");
    check_condition(mxGetM(input[I_METHOD_CODE]) == 1,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a check.scalar.");
    check_condition(mxGetN(input[I_METHOD_CODE]) == 1,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a check.scalar.");
    check_condition(mxIsDouble(input[I_METHOD_CODE]),
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a check.natural.");
    check_condition(fabs(mxGetScalar(input[I_METHOD_CODE]) - floor(mxGetScalar(input[I_METHOD_CODE]))) == 0,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a check.natural.");
    check_condition(mxGetScalar(input[I_METHOD_CODE]) >= 0,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a check.natural.");
    check_condition(mxGetScalar(input[I_METHOD_CODE]) < INT_MAX,
		    "master:InvalidMEXCall","Parameter \"method_code\" is not a check.natural.");
    check_condition((mxGetScalar(input[I_METHOD_CODE]) == L2R_LR) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L2R_L2LOSS_SVC_DUAL) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L2R_L2LOSS_SVC) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L2R_L1LOSS_SVC_DUAL) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L1R_L2LOSS_SVC) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L1R_LR) ||
		    (mxGetScalar(input[I_METHOD_CODE]) == L2R_LR_DUAL),
		    "master:InvalidMEXCall","Parameter \"method_code\" has an invalid value.");
    check_condition(mxGetNumberOfDimensions(input[I_REG_PARAM]) == 2,
		    "master:InvalidMEXCall","Parameter \"reg_param\" is not a check.scalar.");
    check_condition(mxGetM(input[I_REG_PARAM]) == 1,
		    "master:InvalidMEXCall","Parameter \"reg_param\" is not a check.scalar.");
    check_condition(mxGetN(input[I_REG_PARAM]) == 1,
		    "master:InvalidMEXCall","Parameter \"reg_param\" is not a check.scalar.");
    check_condition(mxIsDouble(input[I_REG_PARAM]),
		    "master:InvalidMEXCall","Parameter \"reg_param\" is not a check.number.");
    check_condition(mxGetScalar(input[I_REG_PARAM]) > 0,
		    "master:InvalidMEXCall","Parameter \"reg_param\" is not strictly positive.");
    check_condition(mxGetNumberOfDimensions(input[I_NUM_THREADS]) == 2,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a check.scalar.");
    check_condition(mxGetM(input[I_NUM_THREADS]) == 1,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a check.scalar.");
    check_condition(mxGetN(input[I_NUM_THREADS]) == 1,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a check.scalar.");
    check_condition(mxIsDouble(input[I_NUM_THREADS]),
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a check.natural.");
    check_condition(fabs(mxGetScalar(input[I_NUM_THREADS]) - floor(mxGetScalar(input[I_NUM_THREADS]))) == 0,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a check.natural.");
    check_condition(mxGetScalar(input[I_NUM_THREADS]) >= 0,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a check.natural.");
    check_condition(mxGetScalar(input[I_NUM_THREADS]) < INT_MAX,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not a check.natural.");
    check_condition(input[I_NUM_THREADS] > 0,
		    "master:InvalidMEXCall","Parameter \"num_threads\" is not strictly positive.");
    check_condition(mxGetNumberOfDimensions(input[I_LOGGER]) == 2,
		    "master:InvalidMEXCall","Parameter \"logger\" is not a check.scalar.");
    check_condition(mxGetM(input[I_LOGGER]) == 1,
		    "master:InvalidMEXCall","Parameter \"logger\" is not a check.scalar.");
    check_condition(mxGetN(input[I_LOGGER]) == 1,
		    "master:InvalidMEXCall","Parameter \"logger\" is not a check.scalar.");
    check_condition(strcmp(mxGetClassName(input[I_LOGGER]),"logging.logger") == 0,
		    "master:InvalidMEXCall","Parameter \"logger\" is not a check.logging_logger.");
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

    classifiers_count = classes_count;

    logger_beg_node(local_logger,"Parallel training via \"liblinear\" in One-vs-All fashion");

    logger_beg_node(local_logger,"Passed configuration");

    logger_message(local_logger,"Train sample count: %d",train_sample_count);
    logger_message(local_logger,"Train sample geometry: %d",train_sample_geometry);
    logger_message(local_logger,"Classes count: %d",classes_count);
    logger_message(local_logger,"Method: %s",METHOD_CODE_TO_STRING[method_code]);
    logger_message(local_logger,"Regularization parameter: %.3f",reg_param);
    logger_message(local_logger,"Number of worker threads: %d",num_threads);
    logger_message(local_logger,"Classifiers count: %d",classifiers_count);

    logger_end_node(local_logger);

    /* Count non-null entries for each instance. */

    logger_message(local_logger,"Computing some sample statistics");

    non_null_counts = (mwSize*)mxCalloc(train_sample_count,sizeof(mwSize));
    memset(non_null_counts,0,train_sample_count * sizeof(mwSize));
    non_null_full_count = 0;

    for (ii = 0; ii < train_sample_count; ii++) {
	idx_base = ii * train_sample_geometry;

	for (jj = 0; jj < train_sample_geometry; jj++) {
	    if (train_sample[idx_base + jj] != 0) {
		non_null_counts[ii] = non_null_counts[ii] + 1;
		non_null_full_count = non_null_full_count + 1;
	    }
	}
    }

    logger_beg_node(local_logger,"Sample statistics");

    logger_message(local_logger,"Average number of non-null entries: %d",int(double(non_null_full_count) / double(train_sample_count)));
    logger_message(local_logger,"Total number of non-null entries: %d",non_null_full_count);

    logger_end_node(local_logger);

    /*  Build "prob" problem structure. */

    logger_message(local_logger,"Building problem info structure.");

    prob.l = (int)train_sample_count;
    prob.n = (int)train_sample_geometry + 1;
    prob.y = labels_idx;
    prob.x = (struct feature_node**)mxCalloc(train_sample_count,sizeof(struct feature_node*));
    prob.bias = 1;
    prob_x_t = (struct feature_node*)mxCalloc(non_null_full_count + 2*train_sample_count,sizeof(struct feature_node));

    prob.x[0] = &prob_x_t[0];

    for (ii = 1; ii < train_sample_count; ii++) {
	prob.x[ii] = prob.x[ii - 1] + non_null_counts[ii - 1] + 2;
    }

    logger_beg_node(local_logger,"Problem structure [Sanity Check]");

    logger_message(local_logger,"Number of instances: %d",prob.l);
    logger_message(local_logger,"Number of features: %d",prob.n);
    logger_message(local_logger,"Bias: %.3f",prob.bias);

    logger_end_node(local_logger);

    /* Fill sparse array "prob.x" with data from full array "train_sample". */

    logger_message(local_logger,"Copying data to \"liblinear\" format.");

    current_feature_counts = (mwSize*)mxCalloc(train_sample_count,sizeof(mwSize));
    memset(current_feature_counts,0,train_sample_count);

    for (ii = 0; ii < train_sample_count; ii++) {
	idx_base = ii * train_sample_geometry;

	for (jj = 0; jj < train_sample_geometry; jj++) {
    	    if (train_sample[idx_base + jj] != 0) {
    		prob.x[ii][current_feature_counts[ii]].index = (int)jj + 1;
    		prob.x[ii][current_feature_counts[ii]].value = train_sample[idx_base + jj];
    		current_feature_counts[ii] = current_feature_counts[ii] + 1;
    	    }
    	}
    }

    /* Add bias terms. */

    for (ii = 0; ii < train_sample_count; ii++) {
    	prob.x[ii][current_feature_counts[ii]].index = (int)train_sample_geometry + 1;
    	prob.x[ii][current_feature_counts[ii]].value = 1;
    }

    /* Fill last element of each sparse sample instance with EOL structure. */

    for (ii = 0; ii < train_sample_count; ii++) {
    	prob.x[ii][current_feature_counts[ii] + 1].index = -1;
    	prob.x[ii][current_feature_counts[ii] + 1].value = 0;
    }

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

    /* Call "check_parameter" to validate our problem and parameters structures. */

    logger_message(local_logger,"Checking problem and parameters info structures.");

    check_error = check_parameter(&prob,&param);
    check_condition(check_error == NULL,"master:NoConvergence",check_error);

    /* Build thread pool. */

    logger_message(local_logger,"Building worker task allocation.");

    task_info = (struct task_info*)mxCalloc(classifiers_count,sizeof(struct task_info));
    task_info_results_weights_t = (double*)mxCalloc(classifiers_count * (train_sample_geometry + 1),sizeof(double));

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	task_info[ii_int].class_all = ii_int + 1;
	task_info[ii_int].train_sample_count = train_sample_count;
	task_info[ii_int].train_sample_geometry = train_sample_geometry;
	task_info[ii_int].prob = &prob;
	task_info[ii_int].param = &param;
	task_info[ii_int].results_weights = task_info_results_weights_t + ii_int * (train_sample_geometry + 1);
    }

    /* Start worker threads. */

    logger_message(local_logger,"Starting parallel training of classifiers.");

    run_workers(num_threads,(task_fn_t)do_task,classifiers_count,task_info,sizeof(struct task_info));

    logger_message(local_logger,"Finished parallel training of classifiers.");

    logger_end_node(local_logger);

    /* Build "output". */

    output[O_WEIGHTS] = mxCreateDoubleMatrix(0,0,mxREAL);
    mxSetPr(output[O_WEIGHTS],task_info_results_weights_t);
    mxSetM(output[O_WEIGHTS],train_sample_geometry + 1);
    mxSetN(output[O_WEIGHTS],classifiers_count);

    /* Free memory. */

    mxFree(task_info);
    mxFree(current_feature_counts);
    mxFree(prob_x_t);
    mxFree(prob.x);
    mxFree(non_null_counts);
    mxDestroyArray(local_logger);
}
