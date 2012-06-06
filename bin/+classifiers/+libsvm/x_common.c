#include <stdio.h>
#include <stdarg.h>
#include <errno.h>

#include <pthread.h>
#include <unistd.h>

#include "x_common.h"

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

struct worker_info {
    int          id;
    void       (*task_fn)(void*);
    int          task_buffer_count;
    const void*  task_buffer;
    int          task_buffer_el_size;
};

static void*
do_work(
    void*  worker_info_p) {
    struct worker_info*  worker_info;
    int                  unused;
    int                  pthread_op_res;
    int                  ii;

    pthread_op_res = pthread_setcancelstate(PTHREAD_CANCEL_ENABLE,&unused);
    check_condition(pthread_op_res == 0,"master:SystemError","Could not set thread cancel state!");
    pthread_op_res = pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS,&unused);
    check_condition(pthread_op_res == 0,"master:SystemError","Could not set thread cancel type!");

    worker_info = (struct worker_info*)worker_info_p;

    for (ii = 0; ii < worker_info->task_buffer_count; ii++) {
	worker_info->task_fn((char*)worker_info->task_buffer + ii * worker_info->task_buffer_el_size);
    }

    pthread_exit(NULL);
}

void
run_workers(
    int           num_workers,
    void        (*task_fn)(void*),
    int           task_buffer_count,
    void*         task_buffer,
    int           task_buffer_el_size,
    unsigned int  max_wait_seconds) {
    struct worker_info*  worker_info;
    pthread_t*           thread_handles;
    int                  sleep_res;
    int                  pthread_op_res;
    int                  finished_workers;
    int                  ii_int;

    worker_info = (struct worker_info*)calloc(num_workers,sizeof(struct worker_info));

    for (ii_int = 0; ii_int < num_workers; ii_int++) {
	worker_info[ii_int].id = ii_int;
	worker_info[ii_int].task_fn = task_fn;
	worker_info[ii_int].task_buffer_count = 0;
	worker_info[ii_int].task_buffer = NULL;
	worker_info[ii_int].task_buffer_el_size = task_buffer_el_size;
    }

    for (ii_int = 0; ii_int < task_buffer_count; ii_int++) {
	worker_info[ii_int % num_workers].task_buffer_count += 1;
    }

    worker_info[0].task_buffer = task_buffer;

    for (ii_int = 1; ii_int < num_workers; ii_int++) {
	worker_info[ii_int].task_buffer = (void*)((char*)worker_info[ii_int - 1].task_buffer + worker_info[ii_int - 1].task_buffer_count * task_buffer_el_size);
    }

    thread_handles = (pthread_t*)calloc(num_workers,sizeof(pthread_t));

    for (ii_int = 0; ii_int < num_workers; ii_int++) {
    	pthread_op_res = pthread_create(&thread_handles[ii_int],NULL,do_work,&(worker_info[ii_int]));
    	check_condition(pthread_op_res == 0,"master:SystemError","Could not create thread.");
    }

    sleep_res = max_wait_seconds;

    while (sleep_res != 0) {
	sleep_res = sleep(sleep_res);
    }

    finished_workers = 0;

    for (ii_int = 0; ii_int < num_workers; ii_int++) {
    	pthread_op_res = pthread_tryjoin_np(thread_handles[ii_int],NULL);
    	check_condition(pthread_op_res == 0 || pthread_op_res == EBUSY,"master:SystemError","Could not join thread.");

	if (pthread_op_res == 0) {
	    finished_workers = finished_workers + 1;
	} else {
	    pthread_op_res = pthread_cancel(thread_handles[ii_int]);
	    check_condition(pthread_op_res == 0,"master:SystemError","Could not cancel thread.");
	}
    }

    free(thread_handles);
    free(worker_info);

    check_condition(finished_workers == num_workers,"master:NoConvergence","Could not converge to proper solution in alloted time.");
}
