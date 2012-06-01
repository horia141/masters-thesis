#include <string.h>
#include <math.h>

#include <pthread.h>

#include "mex.h"

#include "svm.h"

#include "x_defines.h"
#include "x_common.h"

enum output_decoder {
    O_SUPPORT_VECTORS_COUNT  = 0,
    O_SUPPORT_VECTORS        = 1,
    O_COEFFS                 = 2,
    O_RHOS                   = 3,
    O_PROB_AS                = 4,
    O_PROB_BS                = 5,
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

struct task_info {
    int                          class_1;
    int                          class_2;
    mwSize                       train_sample_count;
    mwSize                       train_sample_geometry;
    const double*                train_sample;
    const double*                labels_idx;
    const struct svm_parameter*  param;
    int                          results_sv_count;
    int                          results_sv_count_c1;
    int                          results_sv_count_c2;
    struct svm_node**            results_sv;
    struct svm_node*             results_sv_t;
    double*                      results_sv_coeff;
    double                       results_sv_rho;
    double                       results_sv_prob_a;
    double                       results_sv_prob_b;
};

static void
do_task(
    struct task_info*  task_info) {
    struct svm_problem   local_prob;
    struct svm_node*     local_prob_x_t;
    struct svm_model*    result_model;
    const char*          check_error;
    mwSize               class1_count;
    mwSize*              class1_found_index;
    mwSize               class1_current_index;
    mwSize*              class1_non_null_counts;
    mwSize               class1_non_null_full_count;
    mwSize*              class1_current_feature_counts;
    mwSize               class2_count;
    mwSize*              class2_found_index;
    mwSize               class2_current_index;
    mwSize*              class2_non_null_counts;
    mwSize               class2_non_null_full_count;
    mwSize*              class2_current_feature_counts;
    mwSize               sv_non_null_count;
    mwSize               sv_current_count;
    mwSize               ii;
    mwSize               jj;
    mwSize               idx_base;
    int                  ii_int;
    int                  jj_int;

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
    local_prob.y = (double*)calloc(class1_count + class2_count,sizeof(double));
    local_prob.x = (struct svm_node**)calloc(class1_count + class2_count,sizeof(struct svm_node*));
    local_prob_x_t = (struct svm_node*)calloc(class1_non_null_full_count + class1_count +
					      class2_non_null_full_count + class2_count,sizeof(struct svm_node));

    local_prob.y[0] = +1;
    local_prob.x[0] = &local_prob_x_t[0];

    for (ii = 1; ii < class1_count; ii++) {
	local_prob.y[ii] = +1;
	local_prob.x[ii] = local_prob.x[ii - 1] + class1_non_null_counts[ii - 1] + 1;
    }

    local_prob.y[class1_count] = -1;
    local_prob.x[class1_count] = local_prob.x[class1_count - 1] + class1_non_null_counts[class1_count - 1] + 1;

    for (ii = 1; ii < class2_count; ii++) {
	local_prob.y[class1_count + ii] = -1;
	local_prob.x[class1_count + ii] = local_prob.x[class1_count + ii - 1] + class2_non_null_counts[ii - 1] + 1;
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
	local_prob.x[ii][class1_current_feature_counts[ii]].index = -1;
	local_prob.x[ii][class1_current_feature_counts[ii]].value = 0;
    }

    for (ii = 0; ii < class2_count; ii++) {
	local_prob.x[class1_count + ii][class2_current_feature_counts[ii]].index = -1;
	local_prob.x[class1_count + ii][class2_current_feature_counts[ii]].value = 0;
    }

    /* Call "check_parameter" to validate our problem and parameters structures. */

    check_error = svm_check_parameter(&local_prob,task_info->param);
    check_condition(check_error == NULL,"master:NoConvergence",check_error);

    /* Train with local problem. */

    result_model = svm_train(&local_prob,task_info->param);

    sv_non_null_count = 0;

    for (ii_int = 0; ii_int < result_model->l; ii_int++) {
	jj_int = 0;

	while (result_model->SV[ii_int][jj_int].index != -1) {
	    jj_int = jj_int + 1;
	}

	sv_non_null_count = sv_non_null_count + jj_int + 1;
    }

    task_info->results_sv_count = result_model->l;
    task_info->results_sv_count_c1 = result_model->nSV[0];
    task_info->results_sv_count_c2 = result_model->nSV[1];
    task_info->results_sv = (struct svm_node**)calloc(result_model->l,sizeof(struct svm_node*));
    task_info->results_sv_t = (struct svm_node*)calloc(sv_non_null_count,sizeof(struct svm_node));
    task_info->results_sv_coeff = (double*)calloc(result_model->l,sizeof(double));
    memcpy(task_info->results_sv_coeff,result_model->sv_coef[0],result_model->l * sizeof(double));
    task_info->results_sv_rho = result_model->rho[0];
    task_info->results_sv_prob_a = result_model->probA[0];
    task_info->results_sv_prob_b = result_model->probB[0];

    sv_current_count = 0;

    for (ii_int = 0; ii_int < result_model->l; ii_int++) {
	task_info->results_sv[ii_int] = &task_info->results_sv_t[sv_current_count];
	jj_int = 0;

	while (result_model->SV[ii_int][jj_int].index != -1) {
	    task_info->results_sv_t[sv_current_count].index = result_model->SV[ii_int][jj_int].index;
	    task_info->results_sv_t[sv_current_count].value = result_model->SV[ii_int][jj_int].value;
	    sv_current_count = sv_current_count + 1;
	    jj_int = jj_int + 1;
	}

	task_info->results_sv_t[sv_current_count].index = -1;
	task_info->results_sv_t[sv_current_count].value = 0;
	sv_current_count = sv_current_count + 1;
    }

    /* Free memory. */

    svm_free_model_content(result_model);

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
    double*               labels_idx;
    int                   kernel_code;
    double                kernel_param1;
    double                kernel_param2;
    double                reg_param;
    int                   num_threads;
    mxArray*              local_logger;
    int                   classifiers_count;
    struct svm_parameter  param;
    struct task_info*     task_info;
    int                   classifier_index;
    char*                 class1_string;
    char*                 class2_string;
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

    svm_set_print_string_function(printf_wrapper);

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

    classifiers_count = classes_count * (classes_count - 1) / 2;

    logger_beg_node(local_logger,"Parallel training via \"libsvm\" in One-vs-One fashion");

    logger_beg_node(local_logger,"Passed configuration");

    logger_message(local_logger,"Train sample count: %d",train_sample_count);
    logger_message(local_logger,"Train sample geometry: %d",train_sample_geometry);
    logger_message(local_logger,"Classes count: %d",classes_count);
    logger_message(local_logger,"Kernel type: %s",KERNEL_CODE_TO_STRING[kernel_code]);
    logger_message(local_logger,"%s: %f",KERNEL_PARAM1_TO_STRING[kernel_code],kernel_param1);
    logger_message(local_logger,"%s: %f",KERNEL_PARAM2_TO_STRING[kernel_code],kernel_param2);
    logger_message(local_logger,"Regularization parameter: %f",reg_param);
    logger_message(local_logger,"Number of worker threads: %d",num_threads);
    logger_message(local_logger,"Classifiers count: %d",classifiers_count);

    logger_end_node(local_logger);

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

    /* Build thread pool. */

    /* Build thread pool. */

    logger_message(local_logger,"Building worker task allocation.");

    task_info = (struct task_info*)mxCalloc(classifiers_count,sizeof(struct task_info));

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
	    task_info[classifier_index].results_sv_count = -1;
	    task_info[classifier_index].results_sv_count_c1 = -1;
	    task_info[classifier_index].results_sv_count_c2 = -1;
	    task_info[classifier_index].results_sv = NULL;
	    task_info[classifier_index].results_sv_coeff = NULL;
	    task_info[classifier_index].results_sv_prob_a = 0;
	    task_info[classifier_index].results_sv_prob_b = 0;
	    classifier_index = classifier_index + 1;
	}
    }

    /* Starting worker threads. */

    /* Starting worker threads. */

    logger_message(local_logger,"Starting parallel training of classifiers.");

    run_workers(num_threads,(task_fn_t)do_task,classifiers_count,task_info,sizeof(struct task_info));

    logger_message(local_logger,"Finished parallel training of classifiers.");

    /* Print solution summary. */

    logger_beg_node(local_logger,"Solution summary");

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	class1_string = mxArrayToString(mxGetCell(mxGetProperty(input[I_CLASS_INFO],0,"labels"),task_info[ii_int].class_1 - 1));
	class2_string = mxArrayToString(mxGetCell(mxGetProperty(input[I_CLASS_INFO],0,"labels"),task_info[ii_int].class_2 - 1));
	logger_beg_node(local_logger,"%s-vs-%s",class1_string,class2_string);

	logger_message(local_logger,"Number of support vectors: %d",task_info[ii_int].results_sv_count);
	logger_message(local_logger,"Number of support vectors for \"%s\": %d",class1_string,task_info[ii_int].results_sv_count_c1);
	logger_message(local_logger,"Number of support vectors for \"%s\": %d",class2_string,task_info[ii_int].results_sv_count_c2);

	logger_end_node(local_logger);

	mxFree(class2_string);
	mxFree(class1_string);
    }

    logger_end_node(local_logger);

    logger_end_node(local_logger);

    /* Build "output". */

    output[O_SUPPORT_VECTORS_COUNT] = mxCreateCellMatrix(1,classifiers_count);

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	o_support_vectors_count_buf = (double*)mxCalloc(2,sizeof(double));
	o_support_vectors_count_buf[0] = task_info[ii_int].results_sv_count_c1;
	o_support_vectors_count_buf[1] = task_info[ii_int].results_sv_count_c2;

	o_support_vectors_count = mxCreateDoubleMatrix(0,0,mxREAL);
	mxSetPr(o_support_vectors_count,o_support_vectors_count_buf);
	mxSetM(o_support_vectors_count,1);
	mxSetN(o_support_vectors_count,2);

	mxSetCell(output[O_SUPPORT_VECTORS_COUNT],ii_int,o_support_vectors_count);
    }

    output[O_SUPPORT_VECTORS] = mxCreateCellMatrix(1,classifiers_count);

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	o_support_vectors_buf = (double*)mxCalloc(train_sample_geometry * task_info[ii_int].results_sv_count,sizeof(double));

	for (jj_int = 0; jj_int < task_info[ii_int].results_sv_count; jj_int++) {
	    kk_int = 0;

	    while (task_info[ii_int].results_sv[jj_int][kk_int].index != -1) {
		o_index = task_info[ii_int].results_sv[jj_int][kk_int].index - 1;
		o_value = task_info[ii_int].results_sv[jj_int][kk_int].value;

		o_support_vectors_buf[jj_int * train_sample_geometry + o_index] = o_value;
		kk_int = kk_int + 1;
	    }
	}

	o_support_vectors = mxCreateDoubleMatrix(0,0,mxREAL);
	mxSetPr(o_support_vectors,o_support_vectors_buf);
	mxSetM(o_support_vectors,train_sample_geometry);
	mxSetN(o_support_vectors,task_info[ii_int].results_sv_count);

	mxSetCell(output[O_SUPPORT_VECTORS],ii_int,o_support_vectors);
    }

    output[O_COEFFS] = mxCreateCellMatrix(1,classifiers_count);

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	o_coeffs_buf = (double*)mxCalloc(task_info[ii_int].results_sv_count,sizeof(double));
	memcpy(o_coeffs_buf,task_info[ii_int].results_sv_coeff,task_info[ii_int].results_sv_count * sizeof(double));

	o_coeffs = mxCreateDoubleMatrix(0,0,mxREAL);
	mxSetPr(o_coeffs,o_coeffs_buf);
	mxSetM(o_coeffs,1);
	mxSetN(o_coeffs,task_info[ii_int].results_sv_count);

	mxSetCell(output[O_COEFFS],ii_int,o_coeffs);
    }

    output[O_RHOS] = mxCreateCellMatrix(1,classifiers_count);

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	o_rhos = mxCreateDoubleScalar(task_info[ii_int].results_sv_rho);

	mxSetCell(output[O_RHOS],ii_int,o_rhos);
    }

    output[O_PROB_AS] = mxCreateCellMatrix(1,classifiers_count);

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	o_prob_as = mxCreateDoubleScalar(task_info[ii_int].results_sv_prob_a);

	mxSetCell(output[O_PROB_AS],ii_int,o_prob_as);
    }

    output[O_PROB_BS] = mxCreateCellMatrix(1,classifiers_count);

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	o_prob_bs = mxCreateDoubleScalar(task_info[ii_int].results_sv_prob_b);

	mxSetCell(output[O_PROB_BS],ii_int,o_prob_bs);
    }

    /* Free memory. */

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	free(task_info[ii_int].results_sv_t);
	free(task_info[ii_int].results_sv);
	free(task_info[ii_int].results_sv_coeff);
    }

    mxFree(task_info);
    mxDestroyArray(local_logger);
}
