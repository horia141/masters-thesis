#include <stdio.h>
#include <stdarg.h>

#include "x_mex_interface.h"

void
check_condition(
    bool         condition,
    const char*  error_id,
    const char*  message) {
    if (!condition) {
        mexErrMsgIdAndTxt(error_id,message);
    }
}

void
printf_wrapper(const char*  message) {
}

void
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

void
logger_end_node(
    mxArray*  logger) {
    mexCallMATLAB(0,NULL,1,&logger,"end_node");
}

void
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
