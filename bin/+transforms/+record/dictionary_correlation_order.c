#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "mex.h"

#include "x_common.h"

enum output_decoder {
    O_RANKS  = 0,
    OUTPUTS_COUNT
};

enum input_decoder {
    I_SAMPLE      = 0,
    I_MODULATOR   = 1,
    I_NUM_THREADS = 2,
    INPUTS_COUNT
};

static int
component_compare(
    const void* a,
    const void* b) {
    double va;
    double vb;

    va = *(double*)a;
    vb = *(double*)b;

    if (va > vb) {
        return 1;
    } else if (va == vb) {
        return 0;
    } else {
        return -1;
    }
}

struct task_info {
    mwSize         sample_geometry;
    const double*  observation;
    const double*  modulator;
    double*        observation_tmp;
    double*        rank;
};

static void
do_task(
    struct task_info*  task_info) {
    double   component_abs;
    double*  component_foundaddr;
    mwSize   ii;

    for (ii = 0; ii < task_info->sample_geometry; ii++) {
        task_info->observation_tmp[ii] = fabs(task_info->observation[ii]);
    }

    qsort(task_info->observation_tmp,task_info->sample_geometry,sizeof(double),component_compare);

    for (ii = 0; ii < task_info->sample_geometry; ii++) {
        component_abs = fabs(task_info->observation[ii]);
        component_foundaddr = (double*)bsearch(&component_abs,task_info->observation_tmp,task_info->sample_geometry,sizeof(double),component_compare);

        if (task_info->observation[ii] > 0) {
            task_info->rank[ii] = task_info->modulator[task_info->sample_geometry - (component_foundaddr - task_info->observation_tmp) - 1];
        } else {
            task_info->rank[ii] = -task_info->modulator[task_info->sample_geometry - (component_foundaddr - task_info->observation_tmp) - 1];
        }
    }
}

void
mexFunction(
    int             output_count,
    mxArray*        output[],
    int             input_count,
    const mxArray*  input[]) {
    mwSize             sample_count;
    mwSize             sample_geometry;
    const double*      sample;
    const double*      modulator;
    int                num_threads;
    double*            sample_tmp;
    double*            ranks;
    struct task_info*  task_info;
    mwSize             ii;

    /* Validate "input" and "output" parameter. */

    /* ASSUME ALL INPUTS ARE CORRECT. */

    /* Extract relevant information from all inputs. */

    sample_count = mxGetN(input[I_SAMPLE]);
    sample_geometry = mxGetM(input[I_SAMPLE]);
    sample = mxGetPr(input[I_SAMPLE]);
    modulator = mxGetPr(input[I_MODULATOR]);
    num_threads = (int)mxGetScalar(input[I_NUM_THREADS]);

    sample_tmp = (double*)mxCalloc(sample_geometry * sample_count,sizeof(double));
    ranks = (double*)mxCalloc(sample_geometry * sample_count,sizeof(double));

    task_info = (struct task_info*)mxCalloc(sample_count,sizeof(struct task_info));

    for (ii = 0; ii < sample_count; ii++) {
        task_info[ii].sample_geometry = sample_geometry;
        task_info[ii].observation = sample + ii * sample_geometry;
        task_info[ii].modulator = modulator;
        task_info[ii].observation_tmp = sample_tmp + ii * sample_geometry;
        task_info[ii].rank = ranks + ii * sample_geometry;
    }

    run_workers(num_threads,(task_fn_t)do_task,(int)sample_count,task_info,sizeof(struct task_info));

    /* Build "output". */

    output[O_RANKS] = mxCreateDoubleMatrix(0,0,mxREAL);
    mxSetPr(output[O_RANKS],ranks);
    mxSetM(output[O_RANKS],sample_geometry);
    mxSetN(output[O_RANKS],sample_count);

    /* Free memory. */

    mxFree(task_info);
    mxFree(sample_tmp);
}
