#include "mex.h"

#include "liblinear/linear.h"

#include "x_mex_interface.h"
#include "x_classifiers_liblinear_defines.h"
#include "task_control.h"

enum output_decoder {
    O_CLASSIFIERS_DECISIONS  = 0,
    OUTPUTS_COUNT
};

enum input_decoder {
    I_SAMPLE       = 0,
    I_WEIGHTS      = 1,
    I_METHOD_CODE  = 2,
    I_REG_PARAM    = 3,
    I_NUM_WORKERS  = 4,
    INPUTS_COUNT
};

struct global_info {
    size_t        geometry;
    int           classifiers_count;
    const model*  local_models;
};

struct task_info {
    double*        o_decisions;
    size_t         observation_count;
    const double*  observation_pr;
    const size_t*  observation_ir;
};

static void
do_task(
    size_t                     id,
    const struct global_info*  global_info,
    struct global_vars*        global_vars,
    size_t                     task_info_count,
    struct task_info*          task_info) {
    struct feature_node*  observation_features;
    size_t                ii;
    size_t                jj;
    int                   kk;

    observation_features = (struct feature_node*)malloc((global_info->geometry + 2) * sizeof(struct feature_node));

    for (ii = 0; ii < task_info_count; ii++) {
        for (jj = 0; jj < task_info[ii].observation_count; jj++) {
            observation_features[jj].index = (int)task_info[ii].observation_ir[jj] + 1;
            observation_features[jj].value = task_info[ii].observation_pr[jj];
        }

        observation_features[task_info[ii].observation_count + 0].index = global_info->geometry + 1;
        observation_features[task_info[ii].observation_count + 0].value = 1;
        observation_features[task_info[ii].observation_count + 1].index = -1;
        observation_features[task_info[ii].observation_count + 1].value = 0;

        for (kk = 0; kk < global_info->classifiers_count; kk++) {
            predict_values(&global_info->local_models[kk],observation_features,&task_info[ii].o_decisions[kk]);
        }
    }

    free(observation_features);
}

void
mexFunction(
    int             output_count,
    mxArray*        output[],
    int             input_count,
    const mxArray*  input[]) {
    size_t                geometry;
    size_t                sample_count;
    const double*         sample_pr;
    const size_t*         sample_ir;
    const size_t*         sample_jc;
    int                   classifiers_count;
    double*               weights;
    int                   method_code;
    double                reg_param;
    int                   num_workers;
    double*               classifiers_decisions;
    int                   local_model_stub[] = {1,2};
    struct model*         local_models;
    struct global_info    global_info;
    struct task_info*     task_info;
    size_t                ii;
    int                   jj;

    /* For proper output in MATLAB we set this to a correct-type wrapper around "mexPrintf". */

    set_print_string_function(printf_wrapper);

    /* Extract relevant information from all inputs. */

    geometry = mxGetM(input[I_SAMPLE]);
    sample_count = mxGetN(input[I_SAMPLE]);
    sample_pr = mxGetPr(input[I_SAMPLE]);
    sample_ir = mxGetIr(input[I_SAMPLE]);
    sample_jc = mxGetJc(input[I_SAMPLE]);
    classifiers_count = (int)mxGetN(input[I_WEIGHTS]);
    weights = mxGetPr(input[I_WEIGHTS]);
    method_code = (int)mxGetScalar(input[I_METHOD_CODE]);
    reg_param = mxGetScalar(input[I_REG_PARAM]);
    num_workers = (int)mxGetScalar(input[I_NUM_WORKERS]);

    /* Rebuild model structures. */

    local_models = (struct model*)mxMalloc(classifiers_count * sizeof(struct model));

    for (jj = 0; jj < classifiers_count; jj++) {
        local_models[jj].param.solver_type = method_code;
        local_models[jj].param.eps = EPS_DEFAULT[method_code];
        local_models[jj].param.C = reg_param;
        local_models[jj].param.nr_weight = 0;
        local_models[jj].param.weight_label = NULL;
        local_models[jj].param.weight = NULL;
        local_models[jj].param.p = 0;
        local_models[jj].nr_class = 2;
        local_models[jj].nr_feature = (int)geometry + 1;
        local_models[jj].w = weights + jj * (geometry + 1);
        local_models[jj].label = local_model_stub;
        local_models[jj].bias = 1;
    }

    /* Build task distribution information. */

    global_info.geometry = geometry;
    global_info.classifiers_count = classifiers_count;
    global_info.local_models = local_models;

    task_info = (struct task_info*)mxMalloc(sample_count * sizeof(struct task_info));
    classifiers_decisions = (double*)mxMalloc(sample_count * classifiers_count * sizeof(double));

    for (ii = 0; ii < sample_count; ii++) {
        task_info[ii].o_decisions = classifiers_decisions + ii * classifiers_count;
        task_info[ii].observation_count = sample_jc[ii + 1] - sample_jc[ii];
        task_info[ii].observation_pr = sample_pr + sample_jc[ii];
        task_info[ii].observation_ir = sample_ir + sample_jc[ii];
    }

    /* Run workers and compute output. */

    run_workers_x(&global_info,NULL,sample_count,sizeof(struct task_info),task_info,(task_fn_x_t)do_task,num_workers);

    /* Build "output". */

    output[O_CLASSIFIERS_DECISIONS] = mxCreateDoubleMatrix(0,0,mxREAL);
    mxSetPr(output[O_CLASSIFIERS_DECISIONS],classifiers_decisions);
    mxSetM(output[O_CLASSIFIERS_DECISIONS],classifiers_count);
    mxSetN(output[O_CLASSIFIERS_DECISIONS],sample_count);

    /* Free memory. */

    mxFree(task_info);
}
