#include <string.h>

#include "mex.h"

#include "svm.h"

#include "x_common.h"

enum output_decoder {
    O_CLASSIFIERS_PROBS  = 0,
    OUTPUTS_COUNT
};

enum input_decoder {
    I_SAMPLE                 = 0,
    I_SUPPORT_VECTORS_COUNT  = 1,
    I_SUPPORT_VECTORS        = 2,
    I_COEFFS                 = 3,
    I_RHOS                   = 4,
    I_PROB_AS                = 5,
    I_PROB_BS                = 6,
    I_KERNEL_CODE            = 7,
    I_KERNEL_PARAM1          = 8,
    I_KERNEL_PARAM2          = 9,
    I_REG_PARAM              = 10,
    I_LOGGER                 = 11,
    INPUTS_COUNT
};

const mwSize  LOG_BATCH_CONTROL = 10;

void mexFunction(
    int             output_count,
    mxArray*        output[],
    int             input_count,
    const mxArray*  input[]) {
    mwSize             sample_count;
    mwSize             sample_geometry;
    const double*      sample;
    int                classifiers_count;
    int*               sv_count;
    int*               sv_count_c1;
    int*               sv_count_c2;
    double**           sv;
    double**           sv_coeffs;
    double*            sv_rhos;
    double*            sv_prob_as;
    double*            sv_prob_bs;
    int                kernel_code;
    double             kernel_param1;
    double             kernel_param2;
    double             reg_param;
    mxArray*           local_logger;
    mwSize*            non_null_counts;
    mwSize             non_null_full_count;
    mwSize*            current_feature_counts;
    struct svm_model*  local_models;
    struct svm_node**  local_models_SV_t;
    int                check_prob;
    struct svm_node*   instance_features;
    double             decision_value;
    double*            classifiers_probs;
    mwSize             current_feature_count;
    mwSize             log_batch_size;
    mwSize             ii;
    mwSize             jj;
    mwSize             idx_base;
    mwSize             idx_base_features;
    mwSize             idx_base_decisions;
    int                ii_int;
    int                jj_int;

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
    check_condition(mxGetNumberOfDimensions(input[I_SUPPORT_VECTORS_COUNT]) == 2,
		    "master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.vector.");
    check_condition(mxGetM(input[I_SUPPORT_VECTORS_COUNT]) == 1,
		    "master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.vector.");
    check_condition(mxGetN(input[I_SUPPORT_VECTORS_COUNT]) >= 1,
		    "master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.vector.");
    check_condition(mxIsCell(input[I_SUPPORT_VECTORS_COUNT]),
		    "master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell.");
    for (ii = 0; ii < mxGetN(input[I_SUPPORT_VECTORS_COUNT]); ii++) {
	check_condition(mxGetNumberOfDimensions(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii)) == 2,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.vector components.");
	check_condition(mxGetM(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii)) == 1,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.vector components.");
	check_condition(mxGetN(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii)) >= 1,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.vector components.");
	check_condition(mxGetN(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii)) == 2,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.vector components of length 2.");
	check_condition(mxIsDouble(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii)),
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.natural components.");
	check_condition(fabs(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[0] - floor(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[0])) == 0,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.natural components.");
	check_condition(fabs(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[1] - floor(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[1])) == 0,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.natural components.");
	check_condition(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[0] >= 0,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.natural components.");
	check_condition(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[1] >= 0,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.natural components.");
	check_condition(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[0] < INT_MAX,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.natural components.");
	check_condition(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[1] < INT_MAX,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with tc.natural components.");
	check_condition(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[0] > 0,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with strictly positive components.");
	check_condition(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[1] > 0,
			"master:InvalidMEXCall","Parameter \"support_vectors_count\" is not a tc.cell with strictly positive components.");
    }
    check_condition(mxGetNumberOfDimensions(input[I_SUPPORT_VECTORS]) == 2,
		    "master:InvalidMEXCall","Parameter \"support_vectors\" is not a tc.vector.");
    check_condition(mxGetM(input[I_SUPPORT_VECTORS]) == 1,
		    "master:InvalidMEXCall","Parameter \"support_vectors\" is not a tc.vector.");
    check_condition(mxGetN(input[I_SUPPORT_VECTORS]) >= 1,
		    "master:InvalidMEXCall","Parameter \"support_vectors\" is not a tc.vector.");
    check_condition(mxIsCell(input[I_SUPPORT_VECTORS]),
		    "master:InvalidMEXCall","Parameter \"support_vectors\" is not a tc.cell.");
    for (ii = 0; ii < mxGetN(input[I_SUPPORT_VECTORS]); ii++) {
	check_condition(mxGetNumberOfDimensions(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii)) == 2,
			"master:InvalidMEXCall","Parameter \"support_vectors\" is not a tc.cell with tc.matrix components.");
	check_condition(mxGetM(mxGetCell(input[I_SUPPORT_VECTORS],ii)) >= 1,
			"master:InvalidMEXCall","Parameter \"support_vectors\" is not a tc.cell with tc.matrix components.");
	check_condition(mxGetN(mxGetCell(input[I_SUPPORT_VECTORS],ii)) >= 1,
			"master:InvalidMEXCall","Parameter \"support_vectors\" is not a tc.cell with tc.matrix components.");
	check_condition(mxIsDouble(mxGetCell(input[I_SUPPORT_VECTORS],ii)),
			"master:InvalidMEXCall","Parameter \"support_vectors\" is not a tc.cell with tc.number components.");
    }
    check_condition(mxGetNumberOfDimensions(input[I_COEFFS]) == 2,
		    "master:InvalidMEXCall","Parameter \"coeffs\" is not a tc.vector.");
    check_condition(mxGetM(input[I_COEFFS]) == 1,
		    "master:InvalidMEXCall","Parameter \"coeffs\" is not a tc.vector.");
    check_condition(mxGetN(input[I_COEFFS]) >= 1,
		    "master:InvalidMEXCall","Parameter \"coeffs\" is not a tc.vector.");
    check_condition(mxIsCell(input[I_COEFFS]),
		    "master:InvalidMEXCall","Parameter \"coeffs\" is not a tc.cell.");
    for (ii = 0; ii < mxGetN(input[I_COEFFS]); ii++) {
	check_condition(mxGetNumberOfDimensions(mxGetCell(input[I_COEFFS],ii)) == 2,
			"master:InvalidMEXCall","Parameter \"coeffs\" is not a tc.cell with tc.vector components.");
	check_condition(mxGetM(mxGetCell(input[I_COEFFS],ii)) == 1,
			"master:InvalidMEXCall","Parameter \"coeffs\" is not a tc.cell with tc.vector components.");
	check_condition(mxGetN(mxGetCell(input[I_COEFFS],ii)) >= 1,
			"master:InvalidMEXCall","Parameter \"coeffs\" is not a tc.cell with tc.vector components.");
	check_condition(mxIsDouble(mxGetCell(input[I_COEFFS],ii)),
			"master:InvalidMEXCall","Parameter \"coeffs\" is not a tc.cell with tc.number components.");
    }
    check_condition(mxGetNumberOfDimensions(input[I_RHOS]) == 2,
		    "master:InvalidMEXCall","Parameter \"rhos\" is not a tc.vector.");
    check_condition(mxGetM(input[I_RHOS]) == 1,
		    "master:InvalidMEXCall","Parameter \"rhos\" is not a tc.vector.");
    check_condition(mxGetN(input[I_RHOS]) >= 1,
		    "master:InvalidMEXCall","Parameter \"rhos\" is not a tc.vector.");
    check_condition(mxIsCell(input[I_RHOS]),
		    "master:InvalidMEXCall","Parameter \"rhos\" is not a tc.cell.");
    for (ii = 0; ii < mxGetN(input[I_RHOS]); ii++) {
	check_condition(mxGetNumberOfDimensions(mxGetCell(input[I_RHOS],ii)) == 2,
			"master:InvalidMEXCall","Parameter \"rhos\" is not a tc.cell with tc.scalar components.");
	check_condition(mxGetM(mxGetCell(input[I_RHOS],ii)) == 1,
			"master:InvalidMEXCall","Parameter \"rhos\" is not a tc.cell with tc.scalar components.");
	check_condition(mxGetN(mxGetCell(input[I_RHOS],ii)) == 1,
			"master:InvalidMEXCall","Parameter \"rhos\" is not a tc.cell with tc.scalar components.");
	check_condition(mxIsDouble(mxGetCell(input[I_RHOS],ii)),
			"master:InvalidMEXCall","Parameter \"rhos\" is not a tc.cell with tc.number components.");
    }
    check_condition(mxGetNumberOfDimensions(input[I_PROB_AS]) == 2,
		    "master:InvalidMEXCall","Parameter \"prob_as\" is not a tc.vector.");
    check_condition(mxGetM(input[I_PROB_AS]) == 1,
		    "master:InvalidMEXCall","Parameter \"prob_as\" is not a tc.vector.");
    check_condition(mxGetN(input[I_PROB_AS]) >= 1,
		    "master:InvalidMEXCall","Parameter \"prob_as\" is not a tc.vector.");
    check_condition(mxIsCell(input[I_PROB_AS]),
		    "master:InvalidMEXCall","Parameter \"prob_as\" is not a tc.cell.");
    for (ii = 0; ii < mxGetN(input[I_PROB_AS]); ii++) {
	check_condition(mxGetNumberOfDimensions(mxGetCell(input[I_PROB_AS],ii)) == 2,
			"master:InvalidMEXCall","Parameter \"prob_as\" is not a tc.cell with tc.scalar components.");
	check_condition(mxGetM(mxGetCell(input[I_PROB_AS],ii)) == 1,
			"master:InvalidMEXCall","Parameter \"prob_as\" is not a tc.cell with tc.scalar components.");
	check_condition(mxGetN(mxGetCell(input[I_PROB_AS],ii)) == 1,
			"master:InvalidMEXCall","Parameter \"prob_as\" is not a tc.cell with tc.scalar components.");
	check_condition(mxIsDouble(mxGetCell(input[I_PROB_AS],ii)),
			"master:InvalidMEXCall","Parameter \"prob_as\" is not a tc.cell with tc.number components.");
    }
    check_condition(mxGetNumberOfDimensions(input[I_PROB_BS]) == 2,
		    "master:InvalidMEXCall","Parameter \"prob_bs\" is not a tc.vector.");
    check_condition(mxGetM(input[I_PROB_BS]) == 1,
		    "master:InvalidMEXCall","Parameter \"prob_bs\" is not a tc.vector.");
    check_condition(mxGetN(input[I_PROB_BS]) >= 1,
		    "master:InvalidMEXCall","Parameter \"prob_bs\" is not a tc.vector.");
    check_condition(mxIsCell(input[I_PROB_BS]),
		    "master:InvalidMEXCall","Parameter \"coeffs\" is not a tc.cell.");
    for (ii = 0; ii < mxGetN(input[I_PROB_BS]); ii++) {
	check_condition(mxGetNumberOfDimensions(mxGetCell(input[I_PROB_BS],ii)) == 2,
			"master:InvalidMEXCall","Parameter \"prob_bs\" is not a tc.cell with tc.scalar components.");
	check_condition(mxGetM(mxGetCell(input[I_PROB_BS],ii)) == 1,
			"master:InvalidMEXCall","Parameter \"prob_bs\" is not a tc.cell with tc.scalar components.");
	check_condition(mxGetN(mxGetCell(input[I_PROB_BS],ii)) == 1,
			"master:InvalidMEXCall","Parameter \"prob_bs\" is not a tc.cell with tc.scalar components.");
	check_condition(mxIsDouble(mxGetCell(input[I_PROB_BS],ii)),
			"master:InvalidMEXCall","Parameter \"prob_bs\" is not a tc.cell with tc.number components.");
    }
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
		    "master:InvalidMEXCall","Too many sample instances for \"libsvm\".");
    check_condition(mxGetM(input[I_SAMPLE]) < INT_MAX - 1,
		    "master:InvalidMEXCall","Too many features for \"libsvm\".");
    check_condition(mxGetN(input[I_SUPPORT_VECTORS_COUNT]) == mxGetN(input[I_SUPPORT_VECTORS]),
		    "master:InvalidMEXCall","Different number of support vectors and support vector counts.");
    check_condition(mxGetN(input[I_SUPPORT_VECTORS_COUNT]) == mxGetN(input[I_COEFFS]),
		    "master:InvalidMEXCall","Different number of support vector coefficients and support vector counts.");
    check_condition(mxGetN(input[I_SUPPORT_VECTORS_COUNT]) == mxGetN(input[I_RHOS]),
		    "master:InvalidMEXCall","Different number of support vector rhos and support vector counts.");
    check_condition(mxGetN(input[I_SUPPORT_VECTORS_COUNT]) == mxGetN(input[I_PROB_AS]),
		    "master:InvalidMEXCall","Different number of support vector prob_as and support vector counts.");
    check_condition(mxGetN(input[I_SUPPORT_VECTORS_COUNT]) == mxGetN(input[I_PROB_BS]),
		    "master:InvalidMEXCall","Different number of support vector prob_bs and support vector counts.");
    for (ii = 0; ii < mxGetN(input[I_SUPPORT_VECTORS]); ii++) {
	check_condition(mxGetN(mxGetCell(input[I_SUPPORT_VECTORS],ii)) < INT_MAX,
			"master:InvalidMEXCall","Too many support vectors for \"libsvm\".");
	check_condition(mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[0] + 
			mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii))[1] ==
			mxGetN(mxGetCell(input[I_SUPPORT_VECTORS],ii)),
			"master:InvalidMEXCall","Different number of support vectors than indicated by \"support_vector_counts\" for a certain classifier.");
	check_condition(mxGetN(mxGetCell(input[I_SUPPORT_VECTORS],ii)) == mxGetN(mxGetCell(input[I_COEFFS],ii)),
			"master:InvalidMEXCall","Different number of support vectors and support vectors coefficients for a certain classifier.");
    }
    for (ii = 0; ii < mxGetN(input[I_SUPPORT_VECTORS]); ii++) {
	check_condition(mxGetM(mxGetCell(input[I_SUPPORT_VECTORS],ii)) == mxGetM(input[I_SAMPLE]),
			"master:InvalidMEXCell","Support vectors and sample are not compatible for a certain classifier.");
    }

    /* For proper output in MATLAB we set this to a correct-type wrapper around "mexPrintf". */

    svm_set_print_string_function(libsvm_mexPrintf_wrapper);

    /* Extract relevant information from all inputs. */

    sample_count = mxGetN(input[I_SAMPLE]);
    sample_geometry = mxGetM(input[I_SAMPLE]);
    sample = mxGetPr(input[I_SAMPLE]);
    classifiers_count = (int)mxGetN(input[I_SUPPORT_VECTORS]);
    sv_count = (int*)mxCalloc(classifiers_count,sizeof(int));
    sv_count_c1 = (int*)mxCalloc(classifiers_count,sizeof(int));
    sv_count_c2 = (int*)mxCalloc(classifiers_count,sizeof(int));
    sv = (double**)mxCalloc(classifiers_count,sizeof(double*));
    sv_coeffs = (double**)mxCalloc(classifiers_count,sizeof(double*));
    sv_rhos = (double*)mxCalloc(classifiers_count,sizeof(double));
    sv_prob_as = (double*)mxCalloc(classifiers_count,sizeof(double));
    sv_prob_bs = (double*)mxCalloc(classifiers_count,sizeof(double));
    kernel_code = (int)mxGetScalar(input[I_KERNEL_CODE]);
    kernel_param1 = mxGetScalar(input[I_KERNEL_PARAM1]);
    kernel_param2 = mxGetScalar(input[I_KERNEL_PARAM2]);
    reg_param = mxGetScalar(input[I_REG_PARAM]);
    local_logger = mxDuplicateArray(input[I_LOGGER]);

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	sv_count[ii_int] = (int)mxGetN(mxGetCell(input[I_SUPPORT_VECTORS],ii_int));
	sv_count_c1[ii_int] = (int)mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii_int))[0];
	sv_count_c2[ii_int] = (int)mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS_COUNT],ii_int))[1];
	sv[ii_int] = mxGetPr(mxGetCell(input[I_SUPPORT_VECTORS],ii_int));
	sv_coeffs[ii_int] = mxGetPr(mxGetCell(input[I_COEFFS],ii_int));
	sv_rhos[ii_int] = mxGetScalar(mxGetCell(input[I_RHOS],ii_int));
	sv_prob_as[ii_int] = mxGetScalar(mxGetCell(input[I_PROB_AS],ii_int));
	sv_prob_bs[ii_int] = mxGetScalar(mxGetCell(input[I_PROB_BS],ii_int));
    }

    logger_beg_node(local_logger,"Classification via \"libsvm\"");

    logger_beg_node(local_logger,"Passed configuration");

    logger_message(local_logger,"Sample count: %d",sample_count);
    logger_message(local_logger,"Sample geometry: %d",sample_geometry);
    logger_message(local_logger,"Classifiers count: %d",classifiers_count);
    logger_message(local_logger,"Kernel type: %s",KERNEL_CODE_TO_STRING[kernel_code]);
    logger_message(local_logger,"%s: %f",KERNEL_PARAM1_TO_STRING[kernel_code],kernel_param1);
    logger_message(local_logger,"%s: %f",KERNEL_PARAM2_TO_STRING[kernel_code],kernel_param1);
    logger_message(local_logger,"Regularization parameter: %f",reg_param);

    logger_end_node(local_logger);

    /* Rebuild model structures. */

    logger_beg_node(local_logger,"Rebuilding models");

    local_models = (struct svm_model*)mxCalloc(classifiers_count,sizeof(struct svm_model));
    local_models_SV_t = (struct svm_node**)mxCalloc(classifiers_count,sizeof(struct svm_node*));

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	/* Count non-null entries for each support vector in the current problem. */

	non_null_counts = (mwSize*)mxCalloc(sv_count[ii_int],sizeof(mwSize));
	memset(non_null_counts,0,sv_count[ii_int] * sizeof(mwSize));
	non_null_full_count = 0;

	for (jj_int = 0; jj_int < sv_count[ii_int]; jj_int++) {
	    idx_base = jj_int * sample_geometry;

	    for (ii = 0; ii < sample_geometry; ii++) {
		if (sv[ii_int][idx_base + ii] != 0) {
		    non_null_counts[jj_int] = non_null_counts[jj_int] + 1;
		    non_null_full_count = non_null_full_count + 1;
		}
	    }
	}

	/* Initialize "local_models[ii_int]" fields and fill them as much as possible. */

	local_models[ii_int].param.svm_type = C_SVC;
	local_models[ii_int].param.kernel_type = kernel_code;
	local_models[ii_int].param.degree = (int)kernel_param1;
	local_models[ii_int].param.gamma = kernel_param1;
	local_models[ii_int].param.coef0 = kernel_param2;
	local_models[ii_int].param.cache_size = compute_cache_size((int)sample_count);
	local_models[ii_int].param.eps = EPS_DEFAULT[C_SVC];
	local_models[ii_int].param.C = reg_param;
	local_models[ii_int].param.nr_weight = 0;
	local_models[ii_int].param.weight_label = NULL;
	local_models[ii_int].param.weight = NULL;
	local_models[ii_int].param.nu = 0;
	local_models[ii_int].param.p = 0;
	local_models[ii_int].param.shrinking = 0;
	local_models[ii_int].param.probability = 1;
	local_models[ii_int].nr_class = 2;
	local_models[ii_int].l = sv_count[ii_int];
	local_models[ii_int].SV = (struct svm_node**)mxCalloc(sv_count[ii_int],sizeof(struct svm_node*));
	local_models_SV_t[ii_int] = (struct svm_node*)mxCalloc(non_null_full_count + sv_count[ii_int],sizeof(struct svm_node));
	local_models[ii_int].sv_coef = (double**)mxCalloc(1,sizeof(double*));
	local_models[ii_int].sv_coef[0] = (double*)mxCalloc(sv_count[ii_int],sizeof(double));
	memcpy(local_models[ii_int].sv_coef[0],sv_coeffs[ii_int],sv_count[ii_int] * sizeof(double));
	local_models[ii_int].rho = (double*)mxCalloc(1,sizeof(double));
	local_models[ii_int].rho[0] = sv_rhos[ii_int];
	local_models[ii_int].probA = (double*)mxCalloc(1,sizeof(double));
	local_models[ii_int].probA[0] = sv_prob_as[ii_int];
	local_models[ii_int].probB = (double*)mxCalloc(1,sizeof(double));
	local_models[ii_int].probB[0] = sv_prob_bs[ii_int];
	local_models[ii_int].label = (int*)mxCalloc(2,sizeof(int));
	local_models[ii_int].label[0] = 1;
	local_models[ii_int].label[1] = 2;
	local_models[ii_int].nSV = (int*)mxCalloc(2,sizeof(int));
	local_models[ii_int].nSV[0] = sv_count_c1[ii_int];
	local_models[ii_int].nSV[1] = sv_count_c2[ii_int];
	local_models[ii_int].free_sv = 0;

	local_models[ii_int].SV[0] = &local_models_SV_t[ii_int][0];

	for (jj_int = 1; jj_int < sv_count[ii_int]; jj_int++) {
	    local_models[ii_int].SV[jj_int] = local_models[ii_int].SV[jj_int - 1] + non_null_counts[jj_int - 1] + 1;
	}

	/* Fill sparse array "local_models[ii_int].SV" with data from full array "sv[ii_int]". */

	current_feature_counts = (mwSize*)mxCalloc(sv_count[ii_int],sizeof(mwSize));

	for (jj_int = 0; jj_int < sv_count[ii_int]; jj_int++) {
	    idx_base = jj_int * sample_geometry;

	    for (ii = 0; ii < sample_geometry; ii++) {
		if (sv[ii_int][idx_base + ii] != 0) {
		    local_models[ii_int].SV[jj_int][current_feature_counts[jj_int]].index = (int)ii + 1;
		    local_models[ii_int].SV[jj_int][current_feature_counts[jj_int]].value = sv[ii_int][idx_base + ii];
		    current_feature_counts[jj_int] = current_feature_counts[jj_int] + 1;
		}
	    }
	}

        /* Fill last element of each sparse sample instance with EOL structure. */

	for (jj_int = 0; jj_int < sv_count[ii_int]; jj_int++) {
	    local_models[ii_int].SV[jj_int][current_feature_counts[jj_int]].index = -1;
	    local_models[ii_int].SV[jj_int][current_feature_counts[jj_int]].value = 0;
	}

	/* Check probability model. */

	check_prob = svm_check_probability_model(&local_models[ii_int]);
	check_condition(check_prob == 1,"master:InvalidMEXCall","This should not happen!");

	/* Free loop-locally needed data. */

	mxFree(current_feature_counts);
	mxFree(non_null_counts);
    }

    logger_end_node(local_logger);

    logger_beg_node(local_logger,"Models structure [Sanity Check]");

    logger_beg_node(local_logger,"Common");

    logger_message(local_logger,"SVM type: %s",SVM_TYPE_TO_STRING[local_models[0].param.svm_type]);
    logger_message(local_logger,"Kernel type: %s",KERNEL_CODE_TO_STRING[local_models[0].param.kernel_type]);
    logger_message(local_logger,"Degree: %d",local_models[0].param.degree);
    logger_message(local_logger,"Gamma: %f",local_models[0].param.gamma);
    logger_message(local_logger,"Coef0: %f",local_models[0].param.coef0);
    logger_message(local_logger,"Cache size: %.0fMB",local_models[0].param.cache_size);
    logger_message(local_logger,"Epsilon: %f",local_models[0].param.eps);
    logger_message(local_logger,"Regularization param: %f",local_models[0].param.C);
    logger_message(local_logger,"Number of weight biases: %d",local_models[0].param.nr_weight);
    logger_message(local_logger,"nu-SVR Nu: %f",local_models[0].param.nu);
    logger_message(local_logger,"SVR p: %f",local_models[0].param.p);
    logger_message(local_logger,"Use shrinking: %d",local_models[0].param.shrinking);
    logger_message(local_logger,"Use probability: %d",local_models[0].param.probability);

    logger_end_node(local_logger);

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	logger_beg_node(local_logger,"Classifier %d",ii_int);

	logger_message(local_logger,"Number of support vectors: %d",local_models[ii_int].l);
	logger_message(local_logger,"Number of support vectors for first class: %d",local_models[ii_int].nSV[0]);
	logger_message(local_logger,"Number of support vectors for second class: %d",local_models[ii_int].nSV[1]);

	logger_end_node(local_logger);
    }

    logger_end_node(local_logger);

    /* Classify with each set of support vectors/coeffs/rhos supplied. */

    logger_beg_node(local_logger,"Classifying sample");

    classifiers_probs = (double*)mxCalloc(sample_count * classifiers_count,sizeof(double));
    instance_features = (struct svm_node*)mxCalloc(sample_geometry + 1,sizeof(struct svm_node));
    log_batch_size = (int)ceil((double)sample_count / LOG_BATCH_CONTROL);

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

	instance_features[current_feature_count].index = -1;
	instance_features[current_feature_count].value = 0;

	for (jj_int = 0; jj_int < classifiers_count; jj_int++) {
	    svm_predict_values(&local_models[jj_int],instance_features,&decision_value);
	    classifiers_probs[idx_base_decisions + jj_int] = sigmoid_predict(decision_value,local_models[jj_int].probA[0],local_models[jj_int].probB[0]);
	}
    }

    logger_end_node(local_logger);

    logger_end_node(local_logger);

    /* Build "output". */

    output[O_CLASSIFIERS_PROBS] = mxCreateDoubleMatrix(0,0,mxREAL);
    mxSetPr(output[O_CLASSIFIERS_PROBS],classifiers_probs);
    mxSetM(output[O_CLASSIFIERS_PROBS],classifiers_count);
    mxSetN(output[O_CLASSIFIERS_PROBS],sample_count);

    /* Free memory. */

    for (ii_int = 0; ii_int < classifiers_count; ii_int++) {
	mxFree(local_models[ii_int].nSV);
	mxFree(local_models[ii_int].label);
	mxFree(local_models[ii_int].probB);
	mxFree(local_models[ii_int].probA);
	mxFree(local_models[ii_int].rho);
	mxFree(local_models[ii_int].sv_coef[0]);
	mxFree(local_models[ii_int].sv_coef);
	mxFree(local_models_SV_t[ii_int]);
	mxFree(local_models[ii_int].SV);
    }

    mxFree(instance_features);
    mxFree(local_models_SV_t);
    mxFree(local_models);
    mxDestroyArray(local_logger);
    mxFree(sv_rhos);
    mxFree(sv_coeffs);
    mxFree(sv);
    mxFree(sv_count_c2);
    mxFree(sv_count_c1);
    mxFree(sv_count);
}
