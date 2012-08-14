#ifndef _TASK_CONTROL_H
#define _TASK_CONTROL_H

#include "base_defines.h"

typedef void (*task_fn_x_t)(size_t,const void*,void*,size_t,void*);

extern void  run_workers_x(const void* global_info,void* global_vars,size_t task_info_count,size_t task_info_el_size,void* task_info,task_fn_x_t task_fn,size_t num_workers);

#endif
