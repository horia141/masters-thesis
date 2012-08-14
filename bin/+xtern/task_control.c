#include <stdlib.h>
#include <pthread.h>

#include "task_control.h"

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
    int                  ii;

    worker_info = (struct worker_info*)worker_info_p;

    for (ii = 0; ii < worker_info->task_buffer_count; ii++) {
        worker_info->task_fn((char*)worker_info->task_buffer + ii * worker_info->task_buffer_el_size);
    }

    pthread_exit(NULL);
}

void
run_workers(
    int     num_workers,
    void  (*task_fn)(void*),
    int     task_buffer_count,
    void*   task_buffer,
    int     task_buffer_el_size) {
    struct worker_info*  worker_info;
    pthread_t*           thread_handles;
    int                  ii_int;

    worker_info = (struct worker_info*)malloc(num_workers * sizeof(struct worker_info));

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

    thread_handles = (pthread_t*)malloc(num_workers * sizeof(pthread_t));

    for (ii_int = 0; ii_int < num_workers; ii_int++) {
        pthread_create(&thread_handles[ii_int],NULL,do_work,&(worker_info[ii_int]));
    }

    for (ii_int = 0; ii_int < num_workers; ii_int++) {
        pthread_join(thread_handles[ii_int],NULL);
    }

    free(thread_handles);
    free(worker_info);
}


struct worker_info_x {
    size_t       id;
    task_fn_x_t  task_fn;
    const void*  global_info;
    void*        global_vars;
    size_t       task_info_count;
    void*        task_info;
};

static void*
do_work_x(
    void*  worker_info_p) {
    struct worker_info_x*  worker_info;

    worker_info = (struct worker_info_x*)worker_info_p;
    worker_info->task_fn(worker_info->id,worker_info->global_info,worker_info->global_vars,worker_info->task_info_count,worker_info->task_info);
    pthread_exit(NULL);
}

void
run_workers_x(
    const void*  global_info,
    void*        global_vars,
    size_t       task_info_count,
    size_t       task_info_el_size,
    void*        task_info,
    task_fn_x_t  task_fn,
    size_t       num_workers) {
    struct worker_info_x*  worker_info;
    pthread_t*             thread_handles;
    size_t                 worker_task_allocation;
    size_t                 worker_task_allocation_rem;
    size_t                 ii;

    worker_info = (struct worker_info_x*)malloc(num_workers * sizeof(struct worker_info_x));

    worker_task_allocation = task_info_count / num_workers;
    worker_task_allocation_rem = task_info_count % num_workers;

    for (ii = 0; ii < num_workers; ii++) {
        worker_info[ii].id = ii;
        worker_info[ii].task_fn = task_fn;
        worker_info[ii].global_info = global_info;
	worker_info[ii].global_vars = global_vars;

        if (ii == 0) {
            worker_info[ii].task_info_count = worker_task_allocation;
            worker_info[ii].task_info = task_info;
        } else {
            worker_info[ii].task_info_count = worker_task_allocation;
            worker_info[ii].task_info = (void*)((char*)worker_info[ii - 1].task_info + worker_info[ii - 1].task_info_count * task_info_el_size);
        }

        if (ii < worker_task_allocation_rem) {
            worker_info[ii].task_info_count += 1;
        }
    }

    thread_handles = (pthread_t*)malloc(num_workers * sizeof(pthread_t));

    for (ii = 0; ii < num_workers; ii++) {
        pthread_create(&thread_handles[ii],NULL,do_work_x,&(worker_info[ii]));
    }

    for (ii = 0; ii < num_workers; ii++) {
        pthread_join(thread_handles[ii],NULL);
    }

    free(thread_handles);
    free(worker_info);
}
