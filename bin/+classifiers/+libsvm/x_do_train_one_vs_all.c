#include <string.h>

#include <pthread.h>

#include "mex.h"

#include "svm.h"

#include "x_common.h"

enum output_decoder {
    O_SUPPORT_VECTORS_COUNT     = 0,
    O_SUPPORT_VECTORS           = 1,
    O_COEFFS                    = 2,
    O_RHOS                      = 3,
    O_PROB_AS                   = 4,
    O_PROB_BS                   = 5,
    OUTPUTS_COUNT
};

enum input_decoder {
    I_TRAIN_SAMPLE   = 0,
    I_CLASS_INFO     = 1,
    I_KERNEL_CODE    = 2,
    I_KERNEL_PARAM1  = 3,
    I_KERNEL_PARAM2  = 4,
    I_REG_PARAM      = 5,
    I_NUM_THREADS    = 6,
    I_LOGGER         = 7,
    INPUTS_COUNT
};

struct worker_info {
    int                          id;
    mwSize                       train_sample_count;
    mwSize                       train_sample_geometry;
    const struct svm_problem*    prob;
    const struct svm_parameter*  param;
    int                          task_buffer_count;
    const int*                   task_buffer;
    int*                         results_sv_count_base;
    int*                         results_sv_count_c1_base;
    int*                         results_sv_count_c2_base;
    struct svm_node***           results_sv_base;
    double**                     results_sv_coeff_base;
    double*                      results_sv_rho_base;
    double*                      results_sv_prob_a_base;
    double*                      results_sv_prob_b_base;
};

static void*
train_worker(
    void*  worker_info_p) {
    struct worker_info*   worker_info;
    struct svm_problem    local_prob;
    struct svm_model*     result_model;
    struct svm_node*      temp_features;
    double                temp_label;
    mwSize                first_of_one;
    int                   idx_task;
    mwSize                ii;

    worker_info = (struct worker_info*)worker_info_p;

    local_prob.l = (int)worker_info->train_sample_count;
    local_prob.y = (double*)calloc(worker_info->train_sample_count,sizeof(double));
    local_prob.x = (struct svm_node**)calloc(worker_info->train_sample_count,sizeof(struct svm_node*));

    memcpy(local_prob.x,worker_info->prob->x,worker_info->train_sample_count * sizeof(struct svm_node*));

    for (idx_task = 0; idx_task < worker_info->task_buffer_count; idx_task++) {
	first_of_one = 0;

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

	/* Train the binary classifier and copy the model information into "worker_info". */

	result_model = svm_train(&local_prob,worker_info->param);

	worker_info->results_sv_count_base[idx_task] = result_model->l;
	worker_info->results_sv_count_c1_base[idx_task] = result_model->nSV[0];
	worker_info->results_sv_count_c2_base[idx_task] = result_model->nSV[1];
        worker_info->results_sv_base[idx_task] = (struct svm_node**)calloc(result_model->l,sizeof(struct svm_node**));
	memcpy(worker_info->results_sv_base[idx_task],result_model->SV,result_model->l * sizeof(struct svm_node**));
	worker_info->results_sv_coeff_base[idx_task] = (double*)calloc(result_model->l,sizeof(double));
	memcpy(worker_info->results_sv_coeff_base[idx_task],result_model->sv_coef[0],result_model->l * sizeof(double));
	worker_info->results_sv_rho_base[idx_task] = result_model->rho[0];
	worker_info->results_sv_prob_a_base[idx_task] = result_model->probA[0];
	worker_info->results_sv_prob_b_base[idx_task] = result_model->probB[0];

	/* Revert to original order. */

	temp_features = local_prob.x[0];
	local_prob.x[0] = local_prob.x[first_of_one];
	local_prob.x[first_of_one] = temp_features;

	/* Free memory. */

	svm_free_model_content(result_model);
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
    int                   kernel_code;
    double                kernel_param1;
    double                kernel_param2;
    double                reg_param;
    int                   num_threads;
    mxArray*              local_logger;
    mwSize*               non_null_counts;
    mwSize                non_null_full_count;
    mwSize*               current_feature_counts;
    struct svm_problem    prob;
    struct svm_node*      prob_x_t;
    struct svm_parameter  param;
    const char*           check_error;
    int*                  worker_task_buffers_t;
    int*                  worker_results_sv_count_t;
    int*                  worker_results_sv_count_c1_t;
    int*                  worker_results_sv_count_c2_t;
    struct svm_node***    worker_results_sv_t;
    double**              worker_results_sv_coeff_t;
    double*               worker_results_sv_rho_t;
    double*               worker_results_sv_prob_a_t;
    double*               worker_results_sv_prob_b_t;
    struct worker_info*   worker_info;
    pthread_t*            thread_handles;
    int                   pthread_op_res;
    char*                 class_string;
    double*               o_support_vectors_count_buf;
    mxArray*              o_support_vectors_count;
    double*               o_support_vectors_buf;
    mxArray*              o_support_vectors;
    int                   o_index;
    double                o_value;
    double*               o_coeffs_buf;
    mxArray*              o_coeffs;
    mxArray*              o_rhos;
    mxArray*              o_prob_as;
    mxArray*              o_prob_bs;
    mwSize                ii;
    mwSize                jj;
    mwSize                idx_base;
    int                   ii_int;
    int                   jj_int;
    int                   kk_int;

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
    check_condition(mxGetNumberOfDimensions(input[I_KERNEL_CODE]) == 2,
		    "master:InvalidMEXCall","Parameter \"kernel_code\" is not a tc.scalar.");
    check_condition(mxGetM(input[I_KERNEL_CODE]) == 1,
		    "master:InvalidMEXCall","Parameter \"kernel_code\" is not a tc.scalar.");
    check_condition(mxGetN(input[I_KERNEL_CODE]) == 1,
		    "master:InvalidMEXCall","Parameter \"kernel_code\" is not a tc.scalar.");
    check_condition(mxIsDouble(input[I_KERNEL_CODE]),
		    "master:InvalidMEXCall","Parameter \"kernel_code\" is not a tc.natural.");
    check_condition(fabs(mxGetScalar(input[I_KERNEL_CODE]) - floor(mxGetScalar(input[I_KERNEL_CODE]))) == 0,
		    "master:InvalidMEXCall","Parameter \"kernel_code\" is not a tc.natural.");
    check_condition(mxGetScalar(input[I_KERNEL_CODE]) >= 0,
		    "master:InvalidMEXCall","Parameter \"kernel_code\" is not a tc.natural.");
    check_condition(mxGetScalar(input[I_KERNEL_CODE]) < INT_MAX,
		    "master:InvalidMEXCall","Parameter \"kernel_code\" is not a tc.natural.");
    check_condition((mxGetScalar(input[I_KERNEL_CODE]) == LINEAR) ||
		    (mxGetScalar(input[I_KERNEL_CODE]) == POLY) ||
		    (mxGetScalar(input[I_KERNEL_CODE]) == RBF) ||
		    (mxGetScalar(input[I_KERNEL_CODE]) == SIGMOID),
		    "master:InvalidMEXCall","Parameter \"kernel_code\" has an invalid value.");
    check_condition(mxGetNumberOfDimensions(input[I_KERNEL_PARAM1]) == 2,
		    "master:InvalidMEXCall","Parameter \"kernel_param1\" is not a tc.scalar.");
    check_condition(mxGetM(input[I_KERNEL_PARAM1]) == 1,
		    "master:InvalidMEXCall","Parameter \"kernel_param1\" is not a tc.scalar.");
    check_condition(mxGetN(input[I_KERNEL_PARAM1]) == 1,
		    "master:InvalidMEXCall","Parameter \"kernel_param1\" is not a tc.scalar.");
    check_condition(mxIsDouble(input[I_KERNEL_PARAM1]),
		    "master:InvalidMEXCall","Parameter \"kernel_param1\" is not a tc.number.");
    check_condition(mxGetScalar(input[I_KERNEL_PARAM1]) >= 0,
		    "master:InvalidMEXCall","Parameter \"kernel_param1\" is not positive.");
    check_condition(mxGetNumberOfDimensions(input[I_KERNEL_PARAM2]) == 2,
		    "master:InvalidMEXCall","Parameter \"kernel_param2\" is not a tc.scalar.");
    check_condition(mxGetM(input[I_KERNEL_PARAM2]) == 1,
		    "master:InvalidMEXCall","Parameter \"kernel_param2\" is not a tc.scalar.");
    check_condition(mxGetN(input[I_KERNEL_PARAM2]) == 1,
		    "master:InvalidMEXCall","Parameter \"kernel_param2\" is not a tc.scalar.");
    check_condition(mxIsDouble(input[I_KERNEL_PARAM2]),
		    "master:InvalidMEXCall","Parameter \"kernel_param2\" is not a tc.number.");
    check_condition(mxGetScalar(input[I_KERNEL_PARAM2]) >= 0,
		    "master:InvalidMEXCall","Parameter \"kernel_param2\" is not positive.");
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
		    "master:InvalidMEXCall","Too many sample instances for \"libsvm\".");
    check_condition(mxGetM(input[I_TRAIN_SAMPLE]) < INT_MAX - 1,
		    "master:InvalidMEXCall","Too many features for \"libsvm\".");
    check_condition(mxGetScalar(mxGetProperty(input[I_CLASS_INFO],0,"labels_count")) < INT_MAX,
		    "master:InvalidMEXCall","Too many classes for \"libsvm\".");

    /* For proper output in MATLAB we set this to a correct-type wrapper around "mexPrintf". */

    svm_set_print_string_function(libsvm_mexPrintf_wrapper);

    /* Extract relevant information from all inputs. */

    train_sample_count = mxGetN(input[I_TRAIN_SAMPLE]);
    train_sample_geometry = mxGetM(input[I_TRAIN_SAMPLE]);
    train_sample = mxGetPr(input[I_TRAIN_SAMPLE]);
    classes_count = (int)mxGetScalar(mxGetProperty(input[I_CLASS_INFO],0,"labels_count"));
    labels_idx = mxGetPr(mxGetProperty(input[I_CLASS_INFO],0,"labels_idx"));
    kernel_code = (int)mxGetScalar(input[I_KERNEL_CODE]);
    kernel_param1 = mxGetScalar(input[I_KERNEL_PARAM1]);
    kernel_param2 = mxGetScalar(input[I_KERNEL_PARAM2]);
    reg_param = mxGetScalar(input[I_REG_PARAM]);
    num_threads = (int)mxGetScalar(input[I_NUM_THREADS]);
    local_logger = mxDuplicateArray(input[I_LOGGER]);

    logger_beg_node(local_logger,"Parallel training via \"libsvm\" in One-vs-All fashion");

    logger_beg_node(local_logger,"Passed configuration");

    logger_message(local_logger,"Train sample count: %d",train_sample_count);
    logger_message(local_logger,"Train sample geometry: %d",train_sample_geometry);
    logger_message(local_logger,"Classes count: %d",classes_count);
    logger_message(local_logger,"Kernel type: %s",KERNEL_CODE_TO_STRING[kernel_code]);
    logger_message(local_logger,"%s: %f",KERNEL_PARAM1_TO_STRING[kernel_code],kernel_param1);
    logger_message(local_logger,"%s: %f",KERNEL_PARAM2_TO_STRING[kernel_code],kernel_param2);
    logger_message(local_logger,"Regularization parameter: %f",reg_param);
    logger_message(local_logger,"Number of worker threads: %d",num_threads);

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
    logger_message(local_logger,"Total number of non-Null entries: %d",non_null_full_count);

    logger_end_node(local_logger);

    /* Build "prob" problem structure. */

    logger_message(local_logger,"Building problem info structure.");

    prob.l = (int)train_sample_count;
    prob.y = labels_idx;
    prob.x = (struct svm_node**)mxCalloc(train_sample_count,sizeof(struct svm_node*));
    prob_x_t = (struct svm_node*)mxCalloc(non_null_full_count + train_sample_count,sizeof(struct svm_node));

    prob.x[0] = &prob_x_t[0];

    for (ii = 1; ii < train_sample_count; ii++) {
	prob.x[ii] = prob.x[ii - 1] + non_null_counts[ii - 1] + 1;
    }

    logger_beg_node(local_logger,"Problem structure [Sanity Check]");

    logger_message(local_logger,"Number of instances: %d",prob.l);

    logger_end_node(local_logger);

    /* Fill sparse array "prob.x" with data from full array "train_sample". */

    logger_message(local_logger,"Copying data to \"libsvm\" format.");

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

    /* Fill last element of each sparse sample instance with EOL structure. */

    for (ii = 0; ii < train_sample_count; ii++) {
    	prob.x[ii][current_feature_counts[ii]].index = -1;
    	prob.x[ii][current_feature_counts[ii]].value = 0;
    }

    /* Build "param" parameter structure. */

    logger_message(local_logger,"Building parameters info structure.");

    param.svm_type = C_SVC;
    param.kernel_type = kernel_code;
    param.degree = (int)kernel_param1;
    param.gamma = kernel_param1;
    param.coef0 = kernel_param2;
    param.cache_size = compute_cache_size((int)train_sample_count);
    param.eps = EPS_DEFAULT[C_SVC];
    param.C = reg_param;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.nu = 0;
    param.p = 0;
    param.shrinking = 0;
    param.probability = 1;

    logger_beg_node(local_logger,"Parameters structure [Sanity Check]");

    logger_message(local_logger,"SVM type: %s",SVM_TYPE_TO_STRING[param.svm_type]);
    logger_message(local_logger,"Kernel type: %s",KERNEL_CODE_TO_STRING[param.kernel_type]);
    logger_message(local_logger,"Degree: %d",param.degree);
    logger_message(local_logger,"Gamma: %f",param.gamma);
    logger_message(local_logger,"Coef0: %f",param.coef0);
    logger_message(local_logger,"Cache size: %.0fMB",param.cache_size);
    logger_message(local_logger,"Epsilon: %f",param.eps);
    logger_message(local_logger,"Regularization param: %f",param.C);
    logger_message(local_logger,"Number of weight biases: %d",param.nr_weight);
    logger_message(local_logger,"nu-SVR Nu: %f",param.nu);
    logger_message(local_logger,"SVR p: %f",param.p);
    logger_message(local_logger,"Use shrinking: %d",param.shrinking);
    logger_message(local_logger,"Use probability: %d",param.probability);
    
    logger_end_node(local_logger);

    /* Call "check_parameter" to validate our problem and parameters structures. */

    logger_message(local_logger,"Checking parameters info.");

    check_error = svm_check_parameter(&prob,&param);
    check_condition(check_error == NULL,"master:NoConvergence",check_error);

    /* Build thread pool. */

    logger_message(local_logger,"Building worker task allocation.");

    worker_task_buffers_t = (int*)mxCalloc(classes_count,sizeof(int));

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	worker_task_buffers_t[ii_int] = (unsigned int)ii_int;
    }

    worker_results_sv_count_t = (int*)mxCalloc(classes_count,sizeof(int));
    worker_results_sv_count_c1_t = (int*)mxCalloc(classes_count,sizeof(int));
    worker_results_sv_count_c2_t = (int*)mxCalloc(classes_count,sizeof(int));
    worker_results_sv_t = (struct svm_node***)mxCalloc(classes_count,sizeof(struct svm_node**));
    worker_results_sv_coeff_t = (double**)mxCalloc(classes_count,sizeof(double*));
    worker_results_sv_rho_t = (double*)mxCalloc(classes_count,sizeof(double));
    worker_results_sv_prob_a_t = (double*)mxCalloc(classes_count,sizeof(double));
    worker_results_sv_prob_b_t = (double*)mxCalloc(classes_count,sizeof(double));

    worker_info = (struct worker_info*)mxCalloc(num_threads,sizeof(struct worker_info));

    for (ii_int = 0; ii_int < num_threads; ii_int++) {
	worker_info[ii_int].id = ii_int;
	worker_info[ii_int].train_sample_count = train_sample_count;
	worker_info[ii_int].train_sample_geometry = train_sample_geometry;
	worker_info[ii_int].prob = &prob;
	worker_info[ii_int].param = &param;
	worker_info[ii_int].task_buffer_count = 0;
	worker_info[ii_int].task_buffer = NULL;
	worker_info[ii_int].results_sv_count_base = NULL;
	worker_info[ii_int].results_sv_count_c1_base = NULL;
	worker_info[ii_int].results_sv_count_c2_base = NULL;
	worker_info[ii_int].results_sv_base = NULL;
	worker_info[ii_int].results_sv_coeff_base = NULL;
	worker_info[ii_int].results_sv_rho_base = NULL;
	worker_info[ii_int].results_sv_prob_a_base = NULL;
	worker_info[ii_int].results_sv_prob_b_base = NULL;
    }

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	worker_info[ii_int % num_threads].task_buffer_count += 1;
    }

    worker_info[0].task_buffer = &worker_task_buffers_t[0];
    worker_info[0].results_sv_count_base = &worker_results_sv_count_t[0];
    worker_info[0].results_sv_count_c1_base = &worker_results_sv_count_c1_t[0];
    worker_info[0].results_sv_count_c2_base = &worker_results_sv_count_c2_t[0];
    worker_info[0].results_sv_base = &worker_results_sv_t[0];
    worker_info[0].results_sv_coeff_base = &worker_results_sv_coeff_t[0];
    worker_info[0].results_sv_rho_base = &worker_results_sv_rho_t[0];
    worker_info[0].results_sv_prob_a_base = &worker_results_sv_prob_a_t[0];
    worker_info[0].results_sv_prob_b_base = &worker_results_sv_prob_b_t[0];

    for (ii_int = 1; ii_int < num_threads; ii_int++) {
	worker_info[ii_int].task_buffer = worker_info[ii_int - 1].task_buffer + worker_info[ii_int - 1].task_buffer_count;
	worker_info[ii_int].results_sv_count_base = worker_info[ii_int - 1].results_sv_count_base + worker_info[ii_int - 1].task_buffer_count;
	worker_info[ii_int].results_sv_count_c1_base = worker_info[ii_int - 1].results_sv_count_c1_base + worker_info[ii_int - 1].task_buffer_count;
	worker_info[ii_int].results_sv_count_c2_base = worker_info[ii_int - 1].results_sv_count_c2_base + worker_info[ii_int - 1].task_buffer_count;
	worker_info[ii_int].results_sv_base = worker_info[ii_int - 1].results_sv_base + worker_info[ii_int - 1].task_buffer_count;
	worker_info[ii_int].results_sv_coeff_base = worker_info[ii_int - 1].results_sv_coeff_base + worker_info[ii_int - 1].task_buffer_count;
	worker_info[ii_int].results_sv_rho_base = worker_info[ii_int - 1].results_sv_rho_base + worker_info[ii_int - 1].task_buffer_count;
	worker_info[ii_int].results_sv_prob_a_base = worker_info[ii_int - 1].results_sv_prob_a_base + worker_info[ii_int - 1].task_buffer_count;
	worker_info[ii_int].results_sv_prob_b_base = worker_info[ii_int - 1].results_sv_prob_b_base + worker_info[ii_int - 1].task_buffer_count;
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

    /* Starting worker threads. */

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

    /* Print solution summary. */

    logger_beg_node(local_logger,"Solution summary");

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	class_string = mxArrayToString(mxGetCell(mxGetProperty(input[I_CLASS_INFO],0,"labels"),ii_int));
	logger_beg_node(local_logger,"%s-vs-All",class_string);

	logger_message(local_logger,"Number of support vectors: %d",worker_results_sv_count_t[ii_int]);
	logger_message(local_logger,"Number of support vectors for \"%s\": %d",class_string,worker_results_sv_count_c1_t[ii_int]);
	logger_message(local_logger,"Number of support vectors for \"All\": %d",worker_results_sv_count_c2_t[ii_int]);

	logger_end_node(local_logger);

	mxFree(class_string);
    }

    logger_end_node(local_logger);

    logger_end_node(local_logger);

    /* Build "output". */

    output[O_SUPPORT_VECTORS_COUNT] = mxCreateCellMatrix(1,classes_count);

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	o_support_vectors_count_buf = (double*)mxCalloc(2,sizeof(double));
	o_support_vectors_count_buf[0] = worker_results_sv_count_c1_t[ii_int];
	o_support_vectors_count_buf[1] = worker_results_sv_count_c2_t[ii_int];

	o_support_vectors_count = mxCreateDoubleMatrix(0,0,mxREAL);
	mxSetPr(o_support_vectors_count,o_support_vectors_count_buf);
	mxSetM(o_support_vectors_count,1);
	mxSetN(o_support_vectors_count,2);

	mxSetCell(output[O_SUPPORT_VECTORS_COUNT],ii_int,o_support_vectors_count);
    }

    output[O_SUPPORT_VECTORS] = mxCreateCellMatrix(1,classes_count);

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	o_support_vectors_buf = (double*)mxCalloc(train_sample_geometry * worker_results_sv_count_t[ii_int],sizeof(double));

	for (jj_int = 0; jj_int < worker_results_sv_count_t[ii_int]; jj_int++) {
	    kk_int = 0;

	    while (worker_results_sv_t[ii_int][jj_int][kk_int].index != -1) {
		o_index = worker_results_sv_t[ii_int][jj_int][kk_int].index - 1;
		o_value = worker_results_sv_t[ii_int][jj_int][kk_int].value;

		o_support_vectors_buf[jj_int * train_sample_geometry + o_index] = o_value;
		kk_int = kk_int + 1;
	    }
	}

	o_support_vectors = mxCreateDoubleMatrix(0,0,mxREAL);
	mxSetPr(o_support_vectors,o_support_vectors_buf);
	mxSetM(o_support_vectors,train_sample_geometry);
	mxSetN(o_support_vectors,worker_results_sv_count_t[ii_int]);

	mxSetCell(output[O_SUPPORT_VECTORS],ii_int,o_support_vectors);
    }

    output[O_COEFFS] = mxCreateCellMatrix(1,classes_count);

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	o_coeffs_buf = (double*)mxCalloc(worker_results_sv_count_t[ii_int],sizeof(double));
	memcpy(o_coeffs_buf,worker_results_sv_coeff_t[ii_int],worker_results_sv_count_t[ii_int] * sizeof(double));

	o_coeffs = mxCreateDoubleMatrix(0,0,mxREAL);
	mxSetPr(o_coeffs,o_coeffs_buf);
	mxSetM(o_coeffs,1);
	mxSetN(o_coeffs,worker_results_sv_count_t[ii_int]);

	mxSetCell(output[O_COEFFS],ii_int,o_coeffs);
    }

    output[O_RHOS] = mxCreateCellMatrix(1,classes_count);

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	o_rhos = mxCreateDoubleScalar(worker_results_sv_rho_t[ii_int]);

	mxSetCell(output[O_RHOS],ii_int,o_rhos);
    }

    output[O_PROB_AS] = mxCreateCellMatrix(1,classes_count);

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	o_prob_as = mxCreateDoubleScalar(worker_results_sv_prob_a_t[ii_int]);

	mxSetCell(output[O_PROB_AS],ii_int,o_prob_as);
    }

    output[O_PROB_BS] = mxCreateCellMatrix(1,classes_count);

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	o_prob_bs = mxCreateDoubleScalar(worker_results_sv_prob_b_t[ii_int]);

	mxSetCell(output[O_PROB_BS],ii_int,o_prob_bs);
    }

    /* Free memory. */

    for (ii_int = 0; ii_int < classes_count; ii_int++) {
	free(worker_results_sv_t[ii_int]);
	free(worker_results_sv_coeff_t[ii_int]);
    }

    mxFree(thread_handles);
    mxFree(worker_info);
    mxFree(worker_results_sv_prob_b_t);
    mxFree(worker_results_sv_prob_a_t);
    mxFree(worker_results_sv_rho_t);
    mxFree(worker_results_sv_coeff_t);
    mxFree(worker_results_sv_t);
    mxFree(worker_results_sv_count_c2_t);
    mxFree(worker_results_sv_count_c1_t);
    mxFree(worker_results_sv_count_t);
    mxFree(worker_task_buffers_t);
    mxFree(current_feature_counts);
    mxFree(prob_x_t);
    mxFree(prob.x);
    mxFree(non_null_counts);
    mxDestroyArray(local_logger);
}
