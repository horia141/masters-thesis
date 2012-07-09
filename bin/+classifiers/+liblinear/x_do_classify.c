#include <string.h>
#include <math.h>

#include "mex.h"

#include "linear.h"

#include "x_defines.h"
#include "x_common.h"

enum output_decoder {
    O_CLASSIFIERS_DECISIONS  = 0,
    OUTPUTS_COUNT
};

enum input_decoder {
    I_SAMPLE       = 0,
    I_WEIGHTS      = 1,
    I_METHOD_CODE  = 2,
    I_REG_PARAM    = 3,
    I_NUM_THREADS  = 4,
    I_LOGGER       = 5,
    INPUTS_COUNT
};

struct task_info {
    mwSize         instance_idx;
    mwSize         sample_geometry;
    const double*  instance;
    int            classifiers_count;
    const model*   local_models;
    double*        results_decisions;
};

static void
do_task(
    struct task_info*  task_info) {
    mwSize                current_feature_count;
    struct feature_node*  instance_features;
    mwSize                ii;
    int                   ii_int;

    current_feature_count = 0;
    instance_features = (struct feature_node*)calloc(task_info->sample_geometry + 2,sizeof(struct feature_node));

    for (ii = 0; ii < task_info->sample_geometry; ii++) {
	if (task_info->instance[ii] != 0) {
	    instance_features[current_feature_count].index = (int)ii + 1;
	    instance_features[current_feature_count].value = task_info->instance[ii];
	    current_feature_count = current_feature_count + 1;
	}
    }

    instance_features[current_feature_count].index = (int)task_info->sample_geometry + 1;
    instance_features[current_feature_count].value = 1;

    instance_features[current_feature_count + 1].index = -1;
    instance_features[current_feature_count + 1].value = 0;

    for (ii_int = 0; ii_int < task_info->classifiers_count; ii_int++) {
	predict_values(&task_info->local_models[ii_int],instance_features,&task_info->results_decisions[ii_int]);
    }

    free(instance_features);
}

void
mexFunction(
    int             output_count,
    mxArray*        output[],
    int             input_count,
    const mxArray*  input[]) {
    mwSize                sample_count;
    mwSize                sample_geometry;
    const double*         sample;
    int                   classifiers_count;
    double*               weights;
    int                   method_code;
    double                reg_param;
    int                   num_threads;
    mxArray*              local_logger;
    double*               classifiers_decisions;
    int                   local_model_stub[] = {1,2};
    struct model*         local_models;
    struct task_info*     task_info;
    mwSize                ii;
    int                   ii_int;

    /* Validate "input" and "output" parameter. */

    check_condition(output_count == OUTPUTS_COUNT,
		    "master:InvalidMEXCall","Invalid number of outputs.");
    check_condition(input_count == INPUTS_COUNT,
		    "master:InvalidMEXCall","Invalid number of inputs.");
    check_condition(mxGetNumberOfDimensions(input[I_SAMPLE]) == 2,
		    "master:InvalidMEXCall","Parameter \"sample\" is not a check.dataset_record.");
    check_condition(mxGetM(input[I_SAMPLE]) >= 1,
		    "master:InvalidMEXCall","Parameter \"sample\" is not a check.dataset_record.");
    check_condition(mxGetN(input[I_SAMPLE]) >= 1,
		    "master:InvalidMEXCall","Parameter \"sample\" is not a check.dataset_record.");
    check_condition(mxIsDouble(input[I_SAMPLE]),
		    "master:InvalidMEXCall","Parameter \"sample\" is not a check.dataset_record.");
    check_condition(mxGetNumberOfDimensions(input[I_WEIGHTS]) == 2,
		    "master:InvalidMEXCall","Parameter \"weights\" is not a check.matrix.");
    check_condition(mxGetM(input[I_WEIGHTS]) >= 1,
		    "master:InvalidMEXCall","Parameter \"weights\" is not a check.matrix.");
    check_condition(mxGetN(input[I_WEIGHTS]) >= 1,
		    "master:InvalidMEXCall","Parameter \"weights\" is not a check.matrix.");
    check_condition(mxIsDouble(input[I_WEIGHTS]),
		    "master:InvalidMEXCall","Parameter \"weights\" is not a check.dataset_record.");
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
    check_condition(mxGetN(input[I_SAMPLE]) < INT_MAX,
		    "master:InvalidMEXCall","Too many sample instances for \"liblinear\".");
    check_condition(mxGetM(input[I_SAMPLE]) < INT_MAX - 1,
		    "master:InvalidMEXCall","Too many features for \"liblinear\".");
    check_condition(mxGetM(input[I_SAMPLE]) + 1 == mxGetM(input[I_WEIGHTS]),
		    "master:InvalidMEXCall","Different number of features in \"sample\" and \"weights\".");
    check_condition(mxGetN(input[I_WEIGHTS]) < INT_MAX,
		    "master:InvalidMEXCall","Too many classifiers.");

    /* For proper output in MATLAB we set this to a correct-type wrapper around "mexPrintf". */

    set_print_string_function(printf_wrapper);

    /* Extract relevant information from all inputs. */

    sample_count = mxGetN(input[I_SAMPLE]);
    sample_geometry = mxGetM(input[I_SAMPLE]);
    sample = mxGetPr(input[I_SAMPLE]);
    classifiers_count = (int)mxGetN(input[I_WEIGHTS]);
    weights = mxGetPr(input[I_WEIGHTS]);
    method_code = (int)mxGetScalar(input[I_METHOD_CODE]);
    reg_param = mxGetScalar(input[I_REG_PARAM]);
    num_threads = (int)mxGetScalar(input[I_NUM_THREADS]);
    local_logger = mxDuplicateArray(input[I_LOGGER]);

    logger_beg_node(local_logger,"Classification via \"liblinear\"");

    logger_beg_node(local_logger,"Passed configuration");

    logger_message(local_logger,"Sample count: %d",sample_count);
    logger_message(local_logger,"Sample geometry: %d",sample_geometry);
    logger_message(local_logger,"Classifiers count: %d",classifiers_count);
    logger_message(local_logger,"Method: %s",METHOD_CODE_TO_STRING[method_code]);
    logger_message(local_logger,"Regularization parameter: %.3f",reg_param);
    logger_message(local_logger,"Number of worker threads: %d",num_threads);

    logger_end_node(local_logger);

    /* Rebuild model structures. */

    logger_message(local_logger,"Rebuilding models.");

    local_models = (struct model*)mxCalloc(classifiers_count,sizeof(struct model));

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	local_models[ii_int].param.solver_type = method_code;
	local_models[ii_int].param.eps = EPS_DEFAULT[method_code];
	local_models[ii_int].param.C = reg_param;
	local_models[ii_int].param.nr_weight = 0;
	local_models[ii_int].param.weight_label = NULL;
	local_models[ii_int].param.weight = NULL;
	local_models[ii_int].param.p = 0;
	local_models[ii_int].nr_class = 2;
	local_models[ii_int].nr_feature = (int)sample_geometry + 1;
	local_models[ii_int].w = weights + ii_int * (sample_geometry + 1);
	local_models[ii_int].label = local_model_stub;
	local_models[ii_int].bias = 1;
    }

    logger_beg_node(local_logger,"Models structure [Sanity Check]");

    logger_beg_node(local_logger,"Common");

    logger_message(local_logger,"Solver type: %s",METHOD_CODE_TO_STRING[local_models[0].param.solver_type]);
    logger_message(local_logger,"Epsilon: %f",local_models[0].param.eps);
    logger_message(local_logger,"Regularization param: %f",local_models[0].param.C);
    logger_message(local_logger,"Number of weight biases: %d",local_models[0].param.nr_weight);
    logger_message(local_logger,"SVR p: %f",local_models[0].param.p);
    logger_message(local_logger,"Number of features: %d",local_models[0].nr_feature);

    logger_end_node(local_logger);

    logger_end_node(local_logger);

    /* Build thread pool. */

    logger_message(local_logger,"Building worker task allocation.");

    task_info = (struct task_info*)mxCalloc(sample_count,sizeof(struct task_info));
    classifiers_decisions = (double*)mxCalloc(sample_count * classifiers_count,sizeof(double));

    for (ii = 0; ii < sample_count; ii++) {
	task_info[ii].instance_idx = ii;
	task_info[ii].sample_geometry = sample_geometry;
	task_info[ii].instance = sample + ii * sample_geometry;
	task_info[ii].classifiers_count = classifiers_count;
	task_info[ii].local_models = local_models;
	task_info[ii].results_decisions = classifiers_decisions + ii * classifiers_count;
    }

    /* Start worker threads. */

    logger_message(local_logger,"Starting parallel classification.");

    run_workers(num_threads,(task_fn_t)do_task,(int)sample_count,task_info,sizeof(struct task_info));

    logger_message(local_logger,"Finished parallel classification.");

    logger_end_node(local_logger);

    /* Build "output". */

    output[O_CLASSIFIERS_DECISIONS] = mxCreateDoubleMatrix(0,0,mxREAL);
    mxSetPr(output[O_CLASSIFIERS_DECISIONS],classifiers_decisions);
    mxSetM(output[O_CLASSIFIERS_DECISIONS],classifiers_count);
    mxSetN(output[O_CLASSIFIERS_DECISIONS],sample_count);

    /* Free memory. */

    mxDestroyArray(local_logger);
}
