#include <string.h>
#include <math.h>

#include "mex.h"

#include "svm.h"

#include "x_defines.h"
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
    I_NUM_THREADS            = 11,
    I_MAX_WAIT_SECONDS       = 12,
    I_LOGGER                 = 13,
    INPUTS_COUNT
};

struct task_info {
    mwSize                   instance_idx;
    mwSize                   sample_geometry;
    const double*            instance;
    int                      classifiers_count;
    const struct svm_model*  local_models;
    double*                  results_probs;
};

static void
do_task(
    struct task_info*  task_info) {
    mwSize            current_feature_count;
    struct svm_node*  instance_features;
    double            decision_value;
    mwSize            ii;
    int               ii_int;

    current_feature_count = 0;
    instance_features = (struct svm_node*)calloc(task_info->sample_geometry + 2,sizeof(struct svm_node));

    for (ii = 0; ii < task_info->sample_geometry; ii++) {
	if (task_info->instance[ii] != 0) {
	    instance_features[current_feature_count].index = (int)ii + 1;
	    instance_features[current_feature_count].value = task_info->instance[ii];
	    current_feature_count = current_feature_count + 1;
	}
    }

    instance_features[current_feature_count].index = -1;
    instance_features[current_feature_count].value = 0;

    for (ii_int = 0; ii_int < task_info->classifiers_count; ii_int++) {
	svm_predict_values(&task_info->local_models[ii_int],instance_features,&decision_value);
	/* task_info->results_probs[ii_int] = sigmoid_predict(decision_value,task_info->local_models[ii_int].probA[0],task_info->local_models[ii_int].probB[0]); */
	task_info->results_probs[ii_int] = decision_value;
    }

    free(instance_features);
}

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
    int                num_threads;
    unsigned int       max_wait_seconds;
    mxArray*           local_logger;
    mwSize*            non_null_counts;
    mwSize             non_null_full_count;
    mwSize*            current_feature_counts;
    struct svm_model*  local_models;
    struct svm_node**  local_models_SV_t;
    int                check_prob;
    struct task_info*  task_info;
    double*            classifiers_probs;
    mwSize             ii;
    mwSize             idx_base;
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
    check_condition(mxGetNumberOfDimensions(input[I_MAX_WAIT_SECONDS]) == 2,
		    "master:InvalidMEXCall","Parameter \"max_wait_seconds\" is not a tc.scalar.");
    check_condition(mxGetM(input[I_MAX_WAIT_SECONDS]) == 1,
		    "master:InvalidMEXCall","Parameter \"max_wait_seconds\" is not a tc.scalar.");
    check_condition(mxGetN(input[I_MAX_WAIT_SECONDS]) == 1,
		    "master:InvalidMEXCall","Parameter \"max_wait_seconds\" is not a tc.scalar.");
    check_condition(mxIsDouble(input[I_MAX_WAIT_SECONDS]),
		    "master:InvalidMEXCall","Parameter \"max_wait_seconds\" is not a tc.natural.");
    check_condition(fabs(mxGetScalar(input[I_MAX_WAIT_SECONDS]) - floor(mxGetScalar(input[I_MAX_WAIT_SECONDS]))) == 0,
		    "master:InvalidMEXCall","Parameter \"max_wait_seconds\" is not a tc.natural.");
    check_condition(mxGetScalar(input[I_MAX_WAIT_SECONDS]) >= 0,
		    "master:InvalidMEXCall","Parameter \"max_wait_seconds\" is not a tc.natural.");
    check_condition(mxGetScalar(input[I_MAX_WAIT_SECONDS]) < INT_MAX,
		    "master:InvalidMEXCall","Parameter \"max_wait_seconds\" is not a tc.natural.");
    check_condition(input[I_MAX_WAIT_SECONDS] > 0,
		    "master:InvalidMEXCall","Parameter \"max_wait_seconds\" is not strictly positive.");
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

    svm_set_print_string_function(printf_wrapper);

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
    num_threads = (int)mxGetScalar(input[I_NUM_THREADS]);
    max_wait_seconds = (unsigned int)mxGetScalar(input[I_MAX_WAIT_SECONDS]);
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
    logger_message(local_logger,"Number of worker threads: %d",num_threads);
    logger_message(local_logger,"Maximum wait for convergence: %ds",max_wait_seconds);

    logger_end_node(local_logger);

    /* Rebuild model structures. */

    logger_message(local_logger,"Rebuilding models.");

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
	local_models[ii_int].param.shrinking = 1;
	local_models[ii_int].param.probability = 0;
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

    /* Build thread pool. */

    logger_message(local_logger,"Building worker task allocation.");

    task_info = (struct task_info*)mxCalloc(sample_count,sizeof(struct task_info));
    classifiers_probs = (double*)mxCalloc(sample_count * classifiers_count,sizeof(double));

    for (ii = 0; ii < sample_count; ii++) {
	task_info[ii].instance_idx = ii;
	task_info[ii].sample_geometry = sample_geometry;
	task_info[ii].instance = sample + ii * sample_geometry;
	task_info[ii].classifiers_count = classifiers_count;
	task_info[ii].local_models = local_models;
	task_info[ii].results_probs = classifiers_probs + ii * classifiers_count;
    }

    /* Start worker threads. */

    logger_message(local_logger,"Starting parallel classification.");

    run_workers(num_threads,(task_fn_t)do_task,(int)sample_count,task_info,sizeof(struct task_info),max_wait_seconds);

    logger_message(local_logger,"Finished parallel classification.");

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
