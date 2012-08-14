#ifndef _X_MEX_INTERFACE_H
#define _X_MEX_INTERFACE_H

#include "mex.h"

extern void  check_condition(bool condition,const char* error_id,const char* message);

extern void  printf_wrapper(const char* message);

extern void  logger_beg_node(mxArray* logger,const char* fmt_message,...);
extern void  logger_end_node(mxArray* logger);
extern void  logger_message(mxArray* logger,const char* fmt_message,...);

#endif
