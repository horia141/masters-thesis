#ifndef _X_COMMON_H
#define _X_COMMON_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <stdio.h>
#include <stdarg.h>
#undef _GNU_SOURCE
#else
#include <stdio.h>
#include <stdarg.h>
#endif

static const char*
METHOD_CODE_TO_STRING[] = {
    /* 0: L2R_LR */               "L2-regularized logistic regression in primal form",
    /* 1: L2R_L2LOSS_SVC_DUAL */  "L2-regularized L2-loss support vector classification in dual form",
    /* 2: L2R_L2LOSS_SVC */       "L2-regularized L2-loss support vector classification in primal form",
    /* 3: L2R_L1LOSS_SVC_DUAL */  "L2-regularized L1-loss support vector classification in dual form",
    /* 4: unused */               "Unused",
    /* 5: L1R_L2LOSS_SVC */       "L1-regularized L2-loss support vector classification in primal form",
    /* 6: L1R_LR */               "L1-regularized logistic regression in primal form",
    /* 7: L2R_LR_DUAL */          "L2-regularized logistic regression in dual form"
};

static const double
EPS_DEFAULT[] = {
    /* 0: L2R_LR */               0.01,
    /* 1: L2R_L2LOSS_SVC_DUAL */  0.1,
    /* 2: L2R_L2LOSS_SVC */       0.01,
    /* 3: L2R_L1LOSS_SVC_DUAL */  0.1,
    /* 4: unused */               0,
    /* 5: L1R_L2LOSS_SVC */       0.01,
    /* 6: L1R_LR */               0.01,
    /* 7: L2R_LR_DUAL */          0.1
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
liblinear_mexPrintf_wrapper(
    const char*  message) {
    /*mexPrintf(message);*/
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
