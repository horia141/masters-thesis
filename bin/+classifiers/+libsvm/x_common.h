#ifndef _X_COMMON_H
#define _X_COMMON_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <stdio.h>
#undef _GNU_SOURCE
#else
#include <stdio.h>
#endif

#include <stdarg.h>
#include <math.h>

static const char*
SVM_TYPE_TO_STRING[] = {
    /* 0: C_SVC */  "C-SVC"
};

static const double
EPS_DEFAULT[] = {
    /* 0: C_SVC */ 0.001
};

static const char*
KERNEL_CODE_TO_STRING[] = {
    /* 0: LINEAR */   "Linear",
    /* 1: POLY */     "Polynomial",
    /* 2: RBF */      "Gaussian",
    /* 3: SIGMOID */  "Sigmoid"
};

static const char*
KERNEL_PARAM1_TO_STRING[] = {
    /* 0: LINEAR */   "Unused",
    /* 1: POLY */     "Degree",
    /* 2: RBF */      "Gamma",
    /* 3: SIGMOID */  "Gamma"
};

static const char*
KERNEL_PARAM2_TO_STRING[] = {
    /* 0: LINEAR */   "Unused",
    /* 1: POLY */     "Coeff0",
    /* 2: RBF */      "Unused",
    /* 3: SIGMOID */  "Coeff0"
};

static void
check_condition(
    bool         condition,
    const char*  error_id,
    const char*  message) {
    if (!condition) {
	mexErrMsgIdAndTxt(error_id,message);
    }
}

static void
libsvm_mexPrintf_wrapper(
    const char*  message) {
    /*mexPrintf(message);*/
}

static double
compute_cache_size(
    int  sample_count) {
    return fmax(10,(((double)sample_count * (double)sample_count * sizeof(double)) / (double)(1024 * 1024)));
}

static void
logger_beg_node(
    mxArray*     logger, 
    const char*  fmt_message,...) {
    va_list   extra_list;
    char*     extra_message;
    int       print_op_res;
    mxArray*  message_array;
    mxArray*  call_input[2];

    va_start(extra_list,fmt_message);
    print_op_res = vasprintf(&extra_message,fmt_message,extra_list);
    va_end(extra_list);
    check_condition(print_op_res != -1,"master:SystemError","Could not print with format!");

    message_array = mxCreateString(extra_message);
    call_input[0] = logger;
    call_input[1] = message_array;

    mexCallMATLAB(0,NULL,2,call_input,"beg_node");

    mxDestroyArray(message_array);
    free(extra_message);
}

static void
logger_end_node(
    mxArray*  logger) {
    mexCallMATLAB(0,NULL,1,&logger,"end_node");
}

static void
logger_message(
    mxArray*     logger,
    const char*  fmt_message,...) {
    va_list   extra_list;
    char*     extra_message;
    int       print_op_res;
    mxArray*  message_array;
    mxArray*  call_input[2];

    va_start(extra_list,fmt_message);
    print_op_res = vasprintf(&extra_message,fmt_message,extra_list);
    va_end(extra_list);
    check_condition(print_op_res != -1,"master:SystemError","Could not print with format!");

    message_array = mxCreateString(extra_message);
    call_input[0] = logger;
    call_input[1] = message_array;

    mexCallMATLAB(0,NULL,2,call_input,"message");

    mxDestroyArray(message_array);
    free(extra_message);
}

#endif
