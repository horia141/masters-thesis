#include <string.h>
#include <math.h>

#include <pthread.h>

#include "mex.h"

#include "linear.h"

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

struct worker_info {
    int                id;
    mwSize             train_sample_count;
    mwSize             train_sample_geometry;
    struct problem*    prob;
    struct parameter*  param;
    int                task_buffer_count;
    int*               task_buffer;
    double*            results_weights_base;
};

static void*
train_worker(
    void*  worker_info_p) {
    struct worker_info*   worker_info;
    struct problem        local_prob;
    struct model*         result_model;
    struct feature_node*  temp_features;
    double                temp_label;
    mwSize                first_of_one;
    int                   idx_task;
    mwSize                ii;

    worker_info = (struct worker_info*)worker_info_p;

    local_prob.l = (int)worker_info->train_sample_count;
    local_prob.n = (int)worker_info->train_sample_geometry + 1;
    local_prob.y = (double*)calloc(worker_info->train_sample_count,sizeof(double));
    local_prob.x = (struct feature_node**)calloc(worker_info->train_sample_count,sizeof(struct feature_node*));
    local_prob.bias = worker_info->prob->bias;

    memcpy(local_prob.x,worker_info->prob->x,worker_info->train_sample_count * sizeof(struct feature_node*));

    for (idx_task = 0; idx_task < worker_info->task_buffer_count; idx_task++) {
	/* Separate instances into "class" and "all" groups. */

	for (ii = 0; ii < worker_info->train_sample_count; ii++) {
	    if (worker_info->prob->y[ii] == worker_info->task_buffer[idx_task] + 1) {
		local_prob.y[ii] = +1;
	    } else {
		local_prob.y[ii] = -1;
	    }
	}

	/* Make sure the first instance is of the "+1" class. This is needed so that the surface normal
	   points "towards" the "class" instances and we get sane prediction in "x_do_classify". */

	for (ii = 0; ii < worker_info->train_sample_count; ii++) {
	    if (worker_info->prob->y[ii] == worker_info->task_buffer[idx_task] + 1) {
		temp_features = local_prob.x[0];
		local_prob.x[0] = local_prob.x[ii];
		local_prob.x[ii] = temp_features;

		temp_label = local_prob.y[0];
		local_prob.y[0] = local_prob.y[ii];
		local_prob.y[ii] = temp_label;

		first_of_one = ii;

		break;
	    }
	}

	/* Train the binary classifier and copy the resulting surface normal ("weights") into the results
	   buffer. */

	result_model = train(&local_prob,worker_info->param);
	memcpy(worker_info->results_weights_base + idx_task * (worker_info->train_sample_geometry + 1),result_model->w,(worker_info->train_sample_geometry + 1) * sizeof(double));

	temp_features = local_prob.x[0];
	local_prob.x[0] = local_prob.x[first_of_one];
	local_prob.x[first_of_one] = temp_features;

	free_model_content(result_model);
    }

    free(local_prob.x);
    free(local_prob.y);

    pthread_exit(NULL);
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
    mwSize*               non_null_counts;
    mwSize                non_null_full_count;
    mwSize*               current_feature_counts;
    struct problem        prob;
    struct feature_node*  prob_x_t;
    struct parameter      param;
    const char*           check_error;
    int*                  worker_task_buffers_t;
    double*               worker_results_weights_t;
    struct worker_info*   worker_info;
    pthread_t*            thread_handles;
    int                   pthread_op_res;
    char*                 class_string;
    mwSize                ii;
    mwSize                jj;
    mwSize                idx_base;
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

    logger_beg_node(local_logger,"Parallel training via \"liblinear\" in One-vs-All fashion");

    logger_beg_node(local_logger,"Configuration");

    logger_message(local_logger,"Train Sample Count: %d",train_sample_count);
    logger_message(local_logger,"Train Sample Geometry: %d",train_sample_geometry);
    logger_message(local_logger,"Classes Count: %d",classes_count);
    logger_message(local_logger,"Method: %s",METHOD_CODE_TO_STRING[method_code]);
    logger_message(local_logger,"Regularization Parameter: %f",reg_param);
    logger_message(local_logger,"Number of Worker Threads: %d",num_threads);

    /* Count non-null entries for each instance. */

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

    logger_message(local_logger,"Average Number of Non-Null Entries: %d",int(double(non_null_full_count) / double(train_sample_count)));
    logger_message(local_logger,"Total Number of Non-Null Entries: %d",non_null_full_count);

    logger_end_node(local_logger);

    logger_message(local_logger,"Building problem info and copying sample to \"liblinear\" format.");

    /*  Build "prob" problem structure. */

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

    /* Fill sparse array "prob.x" with data from full array "train_sample". */

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

    logger_message(local_logger,"Building parameters info.");

    /* Build "param" parameter structure. */

    param.solver_type = method_code;
    param.eps = EPS_DEFAULT[method_code];
    param.C = reg_param;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.p = 0;

    /* For proper output in MATLAB we set this to a correct-type wrapper around "mexPrintf".
       Later, we should replace this with calls to the "message" function of a "logger" object. */

    set_print_string_function(liblinear_mexPrintf_wrapper);

    /* Call "check_parameter" to validate our problem and parameters structures. */

    logger_message(local_logger,"Checking parameters info.");

    check_error = check_parameter(&prob,&param);
    check_condition(check_error == NULL,"master:NoConvergence",check_error);

    /* Build thread pool. */

    logger_message(local_logger,"Building worker task allocation.");

    worker_task_buffers_t = (int*)mxCalloc(classes_count,sizeof(int));

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	worker_task_buffers_t[ii_int] = (unsigned int)ii_int;
    }

    worker_results_weights_t = (double*)mxCalloc(classes_count * (train_sample_geometry + 1),sizeof(double));

    worker_info = (struct worker_info*)mxCalloc(num_threads,sizeof(struct worker_info));

    for (ii_int = 0; ii_int < num_threads; ii_int++) {
	worker_info[ii_int].id = ii_int;
	worker_info[ii_int].train_sample_count = train_sample_count;
	worker_info[ii_int].train_sample_geometry = train_sample_geometry;
	worker_info[ii_int].prob = &prob;
	worker_info[ii_int].param = &param;
	worker_info[ii_int].task_buffer_count = 0;
	worker_info[ii_int].task_buffer = NULL;
	worker_info[ii_int].results_weights_base = NULL;
    }

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	worker_info[ii_int % num_threads].task_buffer_count += 1;
    }

    worker_info[0].task_buffer = &worker_task_buffers_t[0];
    worker_info[0].results_weights_base = &worker_results_weights_t[0];

    for (ii_int = 1; ii_int < num_threads; ii_int++) {
	worker_info[ii_int].task_buffer = worker_info[ii_int - 1].task_buffer + worker_info[ii_int - 1].task_buffer_count;
	worker_info[ii_int].results_weights_base = worker_info[ii_int - 1].results_weights_base + 
                                                   worker_info[ii_int - 1].task_buffer_count * (train_sample_geometry + 1);
    }

    logger_beg_node(local_logger,"Worker task allocation");

    for (ii_int = 0; ii_int < num_threads; ii_int++) {
	logger_beg_node(local_logger,"Worker %02d",ii_int);

	for (jj_int = 0; jj_int < worker_info[ii_int].task_buffer_count; jj_int++) {
	    class_string = mxArrayToString(mxGetCell(mxGetProperty(input[I_CLASS_INFO],0,"labels"),worker_info[ii_int].task_buffer[jj_int]));
	    logger_message(local_logger,"%s-vs-All",class_string);
	    mxFree(class_string);
	}

	logger_end_node(local_logger);
    }

    logger_end_node(local_logger);

    logger_message(local_logger,"Starting parallel training of classifiers.");

    thread_handles = (pthread_t*)mxCalloc(num_threads,sizeof(pthread_t));

    for (ii_int = 0; ii_int < num_threads; ii_int++) {
    	pthread_op_res = pthread_create(&thread_handles[ii_int],NULL,train_worker,&(worker_info[ii_int]));
    	check_condition(pthread_op_res == 0,"master:SystemError","Could not create thread.");
    }

    for (ii_int = 0; ii_int < num_threads; ii_int++) {
    	pthread_op_res = pthread_join(thread_handles[ii_int],NULL);
    	check_condition(pthread_op_res == 0,"master:SystemError","Could not join thread.");
    }

    logger_message(local_logger,"Finished parallel training of classifiers.");

    logger_end_node(local_logger);

    /* Build "output". */

    output[O_WEIGHTS] = mxCreateDoubleMatrix(0,0,mxREAL);
    mxSetPr(output[O_WEIGHTS],worker_results_weights_t);
    mxSetM(output[O_WEIGHTS],train_sample_geometry + 1);
    mxSetN(output[O_WEIGHTS],classes_count);

    /* Free memory. */

    mxFree(thread_handles);
    mxFree(worker_info);
    mxFree(worker_task_buffers_t);
    mxFree(prob_x_t);
    mxFree(prob.x);
    mxFree(current_feature_counts);
    mxFree(non_null_counts);
    mxDestroyArray(local_logger);
}
