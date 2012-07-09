#ifndef _X_COMMON_H
#define _X_COMMON_H

#include "mex.h"

typedef void (*task_fn_t)(void*);

extern void  check_condition(bool condition,const char* error_id,const char* message);

extern void  printf_wrapper(const char* message);

extern void  logger_beg_node(mxArray* logger,const char* fmt_message,...);
extern void  logger_end_node(mxArray* logger);
extern void  logger_message(mxArray* logger,const char* fmt_message,...);

extern void  run_workers(int num_workers,void (*task_fn)(void*),int task_buffer_count,void* task_buffer,int task_buffer_el_size);

#endif
