#ifndef _TASK_CONTROL_H
#define _TASK_CONTROL_H

#include "base_defines.h"

typedef void (*task_fn_t)(void*);
typedef void (*task_fn_x_t)(size_t,const void*,void*,size_t,void*);

extern void  run_workers(int num_workers,void (*task_fn)(void*),int task_buffer_count,void* task_buffer,int task_buffer_el_size);
extern void  run_workers_x(const void* global_info,void* global_vars,size_t task_info_count,size_t task_info_el_size,void* task_info,task_fn_x_t task_fn,size_t num_workers);

#endif
