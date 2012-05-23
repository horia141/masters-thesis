#include <string.h>
#include <math.h>

#include "mex.h"

#include "linear.h"

#include "x_common.h"

enum output_decoder {
    O_CLASSES_DECISIONS  = 0,
    OUTPUTS_COUNT
};

enum input_decoder {
    I_SAMPLE       = 0,
    I_WEIGHTS      = 1,
    I_METHOD_CODE  = 2,
    I_REG_PARAM    = 3,
    I_LOGGER       = 4,
    INPUTS_COUNT
};

const mwSize  LOG_BATCH_CONTROL = 10;

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
    mxArray*              local_logger;
    double*               classes_decisions;
    struct model          local_model;
    struct feature_node*  instance_features;
    mwSize                current_feature_count;
    mwSize                log_batch_size;
    mwSize                ii;
    mwSize                jj;
    mwSize                idx_base_features;
    mwSize                idx_base_decisions;
    int                   jj_int;

    /* Validate "input" and "output" parameter. */

    check_condition(output_count == OUTPUTS_COUNT,
		    "master:InvalidMEXCall","Invalid number of outputs.");
    check_condition(input_count == INPUTS_COUNT,
		    "master:InvalidMEXCall","Invalid number of inputs.");
    check_condition(mxGetNumberOfDimensions(input[I_SAMPLE]) == 2,
		    "master:InvalidMEXCall","Parameter \"sample\" is not a tc.dataset_record.");
    check_condition(mxGetM(input[I_SAMPLE]) >= 1,
		    "master:InvalidMEXCall","Parameter \"sample\" is not a tc.dataset_record.");
    check_condition(mxGetN(input[I_SAMPLE]) >= 1,
		    "master:InvalidMEXCall","Parameter \"sample\" is not a tc.dataset_record.");
    check_condition(mxIsDouble(input[I_SAMPLE]),
		    "master:InvalidMEXCall","Parameter \"sample\" is not a tc.dataset_record.");
    check_condition(mxGetNumberOfDimensions(input[I_WEIGHTS]) == 2,
		    "master:InvalidMEXCall","Parameter \"weights\" is not a tc.matrix.");
    check_condition(mxGetM(input[I_WEIGHTS]) >= 1,
		    "master:InvalidMEXCall","Parameter \"weights\" is not a tc.matrix.");
    check_condition(mxGetN(input[I_WEIGHTS]) >= 1,
		    "master:InvalidMEXCall","Parameter \"weights\" is not a tc.matrix.");
    check_condition(mxIsDouble(input[I_WEIGHTS]),
		    "master:InvalidMEXCall","Parameter \"weights\" is not a tc.dataset_record.");
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
    check_condition(mxGetN(input[I_SAMPLE]) < INT_MAX,
		    "master:InvalidMEXCall","Too many sample instances for \"liblinear\".");
    check_condition(mxGetM(input[I_SAMPLE]) < INT_MAX - 1,
		    "master:InvalidMEXCall","Too many features for \"liblinear\".");
    check_condition(mxGetM(input[I_SAMPLE]) + 1 == mxGetM(input[I_WEIGHTS]),
		    "master:InvalidMEXCall","Different number of features in \"sample\" and \"weights\".");
    check_condition(mxGetN(input[I_WEIGHTS]) < INT_MAX,
		    "master:InvalidMEXCall","Too many classifiers.");

    /* Extract relevant information from all inputs. */

    sample_count = mxGetN(input[I_SAMPLE]);
    sample_geometry = mxGetM(input[I_SAMPLE]);
    sample = mxGetPr(input[I_SAMPLE]);
    classifiers_count = (int)mxGetN(input[I_WEIGHTS]);
    weights = mxGetPr(input[I_WEIGHTS]);
    method_code = (int)mxGetScalar(input[I_METHOD_CODE]);
    reg_param = mxGetScalar(input[I_REG_PARAM]);
    local_logger = mxDuplicateArray(input[I_LOGGER]);

    logger_beg_node(local_logger,"Classification via \"liblinear\"");

    logger_beg_node(local_logger,"Configuration");

    logger_message(local_logger,"Sample Count: %d",sample_count);
    logger_message(local_logger,"Sample Geometry: %d",sample_geometry);
    logger_message(local_logger,"Classifiers Count: %d",classifiers_count);
    logger_message(local_logger,"Method: %s",METHOD_CODE_TO_STRING[method_code]);
    logger_message(local_logger,"Regularization Parameter: %f",reg_param);

    logger_end_node(local_logger);

    /* For proper output in MATLAB we set this to a correct-type wrapper around "mexPrintf".
       Later, we should replace this with calls to the "message" function of a "logger" object. */

    set_print_string_function(liblinear_mexPrintf_wrapper);

    /* Classify with each set of weights supplied. */

    classes_decisions = (double*)mxCalloc(sample_count * classifiers_count,sizeof(double));
    instance_features = (struct feature_node*)mxCalloc(sample_geometry + 2,sizeof(struct feature_node));
    log_batch_size = sample_count / LOG_BATCH_CONTROL;

    local_model.param.solver_type = method_code;
    local_model.param.eps = EPS_DEFAULT[method_code];
    local_model.param.C = reg_param;
    local_model.param.nr_weight = 0;
    local_model.param.weight_label = NULL;
    local_model.param.weight = NULL;
    local_model.param.p = 0;
    local_model.nr_class = 2;
    local_model.nr_feature = (int)sample_geometry + 1;
    local_model.w = NULL;
    local_model.label = (int*)mxCalloc(2,sizeof(int));
    local_model.label[0] = 1;
    local_model.label[1] = 2;
    local_model.bias = 1;

    logger_beg_node(local_logger,"Classifying sample");

    for (ii = 0; ii < sample_count; ii++) {
	if (ii % log_batch_size == 0) {
	    logger_message(local_logger,"Instance %d to %d.",(int)ii + 1,(int)fmin((double)ii + (double)log_batch_size - 1,(double)sample_count) + 1);
	}

	idx_base_decisions = ii * classifiers_count;
	current_feature_count = 0;

	for (jj = 0; jj < sample_geometry; jj++) {
	    idx_base_features = ii * sample_geometry;

	    if (sample[idx_base_features + jj] != 0) {
		instance_features[current_feature_count].index = (int)jj + 1;
		instance_features[current_feature_count].value = sample[idx_base_features + jj];
		current_feature_count = current_feature_count + 1;
	    }
	}

	instance_features[current_feature_count].index = (int)sample_geometry + 1;
	instance_features[current_feature_count].value = 1;

	instance_features[current_feature_count + 1].index = -1;
	instance_features[current_feature_count + 1].value = 0;

	for (jj_int = 0; jj_int < classifiers_count; jj_int++) {
	    local_model.w = weights + jj_int * (sample_geometry + 1);
	    predict_values(&local_model,instance_features,&classes_decisions[idx_base_decisions + jj_int]);
	}
    }

    logger_end_node(local_logger);

    logger_end_node(local_logger);

    /* Build "output". */

    output[O_CLASSES_DECISIONS] = mxCreateDoubleMatrix(0,0,mxREAL);
    mxSetPr(output[O_CLASSES_DECISIONS],classes_decisions);
    mxSetM(output[O_CLASSES_DECISIONS],classifiers_count);
    mxSetN(output[O_CLASSES_DECISIONS],sample_count);

    /* Free memory. */

    mxFree(local_model.label);
    mxFree(instance_features);
}
