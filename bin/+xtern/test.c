#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#include "base_defines.h"
#include "latools.h"
#include "coding_methods.h"
#include "image_coder.h"
#include "task_control.h"

struct task_info {
    int  o_result;
    int  value;
};

static void
_do_work(
    struct task_info*  task_info) {
    task_info->o_result = 2 * task_info->value;

    sleep(3);
}

struct global_info_x {
    int  alpha;
    int  beta;
};

struct global_vars_x {
    int              counter;
    pthread_mutex_t  counter_lock;
};

struct task_info_x {
    int  o_result;
    int  value;
};

static void
_do_work_x(
    int                          id,
    const struct global_info_x*  global_info,
    struct global_vars_x*        global_vars,
    size_t                       task_info_count,
    struct task_info_x*          task_info) {
    size_t  ii;

    for (ii = 0; ii < task_info_count; ii++) {
        task_info[ii].o_result = global_info->alpha * task_info[ii].value + global_info->beta;
        sleep(3);

	pthread_mutex_lock(&global_vars->counter_lock);
	global_vars->counter++;
	pthread_mutex_unlock(&global_vars->counter_lock);
    }
}

int
main(
    int     argc,
    char**  argv) {

    printf("Testing \"latools\".\n");

    printf("  Function \"dict_obs_inner_product\".\n");

    {
        double  o_similarities[] = {0,0,0,0};
        size_t  geometry = 3;
        size_t  word_count = 4;
        double  dict[] = {1,0,0,1,0,1,0,1,0,0,1,0};
        double  dict_transp[] = {1,0,0,0,1,0,0,0,1,1,1,0};
        double  observation[] = {1,2,3};

        dict_obs_inner_product(o_similarities,geometry,word_count,dict,dict_transp,observation);

        assert(o_similarities[0] == 1);
        assert(o_similarities[1] == 2);
        assert(o_similarities[2] == 3);
        assert(o_similarities[3] == 3);
    }

    {
        double  o_similarities[] = {0,0,0};
        size_t  geometry = 2;
        size_t  word_count = 3;
        double  dict[] = {1,0,1,0,-1,1};
        double  dict_transp[] = {1,0,0,-1,1,1};
        double  observation[] = {14,-3};

        dict_obs_inner_product(o_similarities,geometry,word_count,dict,dict_transp,observation);

        assert(o_similarities[0] == 14);
        assert(o_similarities[1] == 3);
        assert(o_similarities[2] == 11);
    }

    printf("  Function \"fill_idx_1n\".\n");

    {
        size_t  o_idx[] = {0,0,0,0,0,0,0};
        size_t  count = 7;

        fill_idx_1n(o_idx,count);

        assert(o_idx[0] == 0);
        assert(o_idx[1] == 1);
        assert(o_idx[2] == 2);
        assert(o_idx[3] == 3);
        assert(o_idx[4] == 4);
        assert(o_idx[5] == 5);
        assert(o_idx[6] == 6);
    }

    {
        size_t  o_idx[] = {0,0,0,0,0,0,0};
        size_t  count = 4;

        fill_idx_1n(o_idx,count);

        assert(o_idx[0] == 0);
        assert(o_idx[1] == 1);
        assert(o_idx[2] == 2);
        assert(o_idx[3] == 3);
        assert(o_idx[4] == 0);
        assert(o_idx[5] == 0);
        assert(o_idx[6] == 0);
    }

    printf("  Function \"sort_by_abs_coeffs\".\n");

    {
        double  o_coeffs[] = {1,3,-2,4};
        size_t  o_coeffs_idx[] = {0,1,2,3};
        size_t  count = 4;

        sort_by_abs_coeffs(o_coeffs,o_coeffs_idx,count);

        assert(o_coeffs[0] == 4);
        assert(o_coeffs[1] == 3);
        assert(o_coeffs[2] == -2);
        assert(o_coeffs[3] == 1);
        assert(o_coeffs_idx[0] == 3);
        assert(o_coeffs_idx[1] == 1);
        assert(o_coeffs_idx[2] == 2);
        assert(o_coeffs_idx[3] == 0);
    }

    printf("  Function \"sort_by_idxs\".\n");

    {
        double  o_coeffs[] = {4,3,-2,1};
        size_t  o_coeffs_idx[] = {3,1,2,0};
        size_t  count = 4;

        sort_by_idxs(o_coeffs,o_coeffs_idx,count);

        assert(o_coeffs[0] == 1);
        assert(o_coeffs[1] == 3);
        assert(o_coeffs[2] == -2);
        assert(o_coeffs[3] == 4);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(o_coeffs_idx[2] == 2);
        assert(o_coeffs_idx[3] == 3);
    }

    printf("Testing \"coding_methods\".\n");

    printf("  Function \"correlation\".\n");

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,1,0,-1,1};
        double   dict_transp[] = {1,0,0,-1,1,1};
        double   dict_x_dict_transp[] = {1,0,1,0,1,1,1,1,2};
        size_t   coeffs_count = 2;
        double   observation[] = {4,-3};
        char*    coding_tmps;
        double*  tmp_similarities;
        size_t*  tmp_similarities_idx;

        coding_tmps = malloc(3*sizeof(double) + 3*sizeof(size_t));
        tmp_similarities = (double*)coding_tmps;
        tmp_similarities_idx = (size_t*)((double*)coding_tmps + 3);

        correlation(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeffs_count,NULL,observation,coding_tmps);

        assert(o_coeffs[0] == 4);
        assert(o_coeffs[1] == 3);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(tmp_similarities[0] == 4);
        assert(tmp_similarities[1] == 3);
        assert(tmp_similarities[2] == 1);
        assert(tmp_similarities_idx[0] == 0);
        assert(tmp_similarities_idx[1] == 1);
        assert(tmp_similarities_idx[2] == 2);

        free(coding_tmps);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,1,0,-1,1};
        double   dict_transp[] = {1,0,0,-1,1,1};
        double   dict_x_dict_transp[] = {1,0,1,0,1,1,1,1,2};
        size_t   coeffs_count = 2;
        double   observation[] = {4,3};
        char*    coding_tmps;
        double*  tmp_similarities;
        size_t*  tmp_similarities_idx;

        coding_tmps = malloc(3*sizeof(double) + 3*sizeof(size_t));
        tmp_similarities = (double*)coding_tmps;
        tmp_similarities_idx = (size_t*)((double*)coding_tmps + 3);

        correlation(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeffs_count,NULL,observation,coding_tmps);

        assert(o_coeffs[0] == 4);
        assert(o_coeffs[1] == 7);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 2);
        assert(tmp_similarities[0] == 7);
        assert(tmp_similarities[1] == 4);
        assert(tmp_similarities[2] == -3);
        assert(tmp_similarities_idx[0] == 2);
        assert(tmp_similarities_idx[1] == 0);
        assert(tmp_similarities_idx[2] == 1);

        free(coding_tmps);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,1,0,-1,1};
        double   dict_transp[] = {1,0,0,-1,1,1};
        double   dict_x_dict_transp[] = {1,0,1,0,1,1,1,1,2};
        size_t   coeffs_count = 2;
        double   observation[] = {-4,3};
        char*    coding_tmps;
        double*  tmp_similarities;
        size_t*  tmp_similarities_idx;

        coding_tmps = malloc(3*sizeof(double) + 3*sizeof(size_t));
        tmp_similarities = (double*)coding_tmps;
        tmp_similarities_idx = (size_t*)((double*)coding_tmps + 3);

        correlation(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeffs_count,NULL,observation,coding_tmps);

        assert(o_coeffs[0] == -4);
        assert(o_coeffs[1] == -3);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(tmp_similarities[0] == -4);
        assert(tmp_similarities[1] == -3);
        assert(tmp_similarities[2] == -1);
        assert(tmp_similarities_idx[0] == 0);
        assert(tmp_similarities_idx[1] == 1);
        assert(tmp_similarities_idx[2] == 2);

        free(coding_tmps);
    }

    printf("  Function \"correlation_order\".\n");

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,1,0,-1,1};
        double   dict_transp[] = {1,0,0,-1,1,1};
        double   dict_x_dict_transp[] = {1,0,1,0,1,1,1,1,2};
        size_t   coeffs_count = 2;
        double   modulator[] = {1.11,0.512};
        double   observation[] = {4,-3};
        char*    coding_tmps;
        double*  tmp_similarities;
        size_t*  tmp_similarities_idx;

        coding_tmps = malloc(3*sizeof(double) + 3*sizeof(size_t));
        tmp_similarities = (double*)coding_tmps;
        tmp_similarities_idx = (size_t*)((double*)coding_tmps + 3);

        correlation_order(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeffs_count,modulator,observation,coding_tmps);

        assert(o_coeffs[0] == 1.11);
        assert(o_coeffs[1] == 0.512);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(tmp_similarities[0] == 4);
        assert(tmp_similarities[1] == 3);
        assert(tmp_similarities[2] == 1);
        assert(tmp_similarities_idx[0] == 0);
        assert(tmp_similarities_idx[1] == 1);
        assert(tmp_similarities_idx[2] == 2);

        free(coding_tmps);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,1,0,-1,1};
        double   dict_transp[] = {1,0,0,-1,1,1};
        double   dict_x_dict_transp[] = {1,0,1,0,1,1,1,1,2};
        size_t   coeffs_count = 2;
        double   modulator[] = {1.11,0.512};
        double   observation[] = {4,3};
        char*    coding_tmps;
        double*  tmp_similarities;
        size_t*  tmp_similarities_idx;

        coding_tmps = malloc(3*sizeof(double) + 3*sizeof(size_t));
        tmp_similarities = (double*)coding_tmps;
        tmp_similarities_idx = (size_t*)((double*)coding_tmps + 3);

        correlation_order(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeffs_count,modulator,observation,coding_tmps);

        assert(o_coeffs[0] == 0.512);
        assert(o_coeffs[1] == 1.11);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 2);
        assert(tmp_similarities[0] == 7);
        assert(tmp_similarities[1] == 4);
        assert(tmp_similarities[2] == -3);
        assert(tmp_similarities_idx[0] == 2);
        assert(tmp_similarities_idx[1] == 0);
        assert(tmp_similarities_idx[2] == 1);

        free(coding_tmps);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,1,0,-1,1};
        double   dict_transp[] = {1,0,0,-1,1,1};
        double   dict_x_dict_transp[] = {1,0,1,0,1,1,1,1,2};
        size_t   coeffs_count = 2;
        double   modulator[] = {1.11,0.512};
        double   observation[] = {-4,3};
        char*    coding_tmps;
        double*  tmp_similarities;
        size_t*  tmp_similarities_idx;

        coding_tmps = malloc(3*sizeof(double) + 3*sizeof(size_t));
        tmp_similarities = (double*)coding_tmps;
        tmp_similarities_idx = (size_t*)((double*)coding_tmps + 3);

        correlation_order(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeffs_count,modulator,observation,coding_tmps);

        assert(o_coeffs[0] == -1.11);
        assert(o_coeffs[1] == -0.512);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(tmp_similarities[0] == -4);
        assert(tmp_similarities[1] == -3);
        assert(tmp_similarities[2] == -1);
        assert(tmp_similarities_idx[0] == 0);
        assert(tmp_similarities_idx[1] == 1);
        assert(tmp_similarities_idx[2] == 2);

        free(coding_tmps);
    }

    printf("  Function \"matching_pursuit\".\n");

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeffs_count = 2;
        double   observation[] = {4,-3};
        double   tmp_similarities[] = {0,0,0};

        matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeffs_count,NULL,observation,tmp_similarities);

        assert(fabs(o_coeffs[0] - 4) < 1e-4);
        assert(fabs(o_coeffs[1] - 3) < 1e-4);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(fabs(tmp_similarities[0] - 0) < 1e-4);
        assert(fabs(tmp_similarities[1] - 0) < 1e-4);
        assert(fabs(tmp_similarities[2] - 0) < 1e-4);
    }


    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeffs_count = 2;
        double   observation[] = {4,3};
        double   tmp_similarities[] = {0,0,0};

        matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeffs_count,NULL,observation,tmp_similarities);

        assert(fabs(o_coeffs[0] - 0.5000) < 1e-4);
        assert(fabs(o_coeffs[1] - 4.9497) < 1e-4);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 2);
        assert(fabs(tmp_similarities[0] - 0) < 1e-4);
        assert(fabs(tmp_similarities[1] - 0.4999) < 1e-4);
        assert(fabs(tmp_similarities[2] - -0.3535) < 1e-4);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeffs_count = 2;
        double   observation[] = {-4,3};
        double   tmp_similarities[] = {0,0,0};

        matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeffs_count,NULL,observation,tmp_similarities);

        assert(fabs(o_coeffs[0] - -4) < 1e-4);
        assert(fabs(o_coeffs[1] - -3) < 1e-4);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(fabs(tmp_similarities[0] - 0) < 1e-4);
        assert(fabs(tmp_similarities[1] - 0) < 1e-4);
        assert(fabs(tmp_similarities[2] - 0) < 1e-4);
    }

    printf("Testing \"image_coder\".\n");

    printf("  Function \"code_image_new_geometry\".\n");

    {
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,1) == 78400);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,2) == 19600);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,3) == 8100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,4) == 4900);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,5) == 2500);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,6) == 1600);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,7) == 1600);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,8) == 900);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,10) == 400);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,11) == 400);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,12) == 400);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,13) == 400);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,14) == 400);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,15) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,16) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,17) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,18) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,19) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,20) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,21) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,22) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,23) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,24) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,25) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,26) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,27) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NONE,28) == 100);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,1) == 156800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,2) == 39200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,3) == 16200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,4) == 9800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,5) == 5000);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,6) == 3200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,7) == 3200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,8) == 1800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,10) == 800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,11) == 800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,12) == 800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,13) == 800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,14) == 800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,15) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,16) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,17) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,18) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,19) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,20) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,21) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,22) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,23) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,24) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,25) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,26) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,27) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,NO_SIGN,28) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,1) == 156800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,2) == 39200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,3) == 16200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,4) == 9800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,5) == 5000);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,6) == 3200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,7) == 3200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,8) == 1800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,10) == 800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,11) == 800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,12) == 800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,13) == 800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,14) == 800);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,15) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,16) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,17) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,18) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,19) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,20) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,21) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,22) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,23) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,24) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,25) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,26) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,27) == 200);
	assert(code_image_new_geometry(28,28,IDENTITY,0,0,100,KEEP_SIGN,28) == 200);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,1) == 36864);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,2) == 9216);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,3) == 4096);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,4) == 2304);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,5) == 1024);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,6) == 1024);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,7) == 256);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,8) == 256);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,9) == 256);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,10) == 256);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,11) == 256);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NONE,12) == 256);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,1) == 73728);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,2) == 18432);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,3) == 8192);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,4) == 4608);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,5) == 2048);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,6) == 2048);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,7) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,8) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,9) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,10) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,11) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,NO_SIGN,12) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,1) == 73728);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,2) == 18432);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,3) == 8192);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,4) == 4608);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,5) == 2048);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,6) == 2048);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,7) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,8) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,9) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,10) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,11) == 512);
	assert(code_image_new_geometry(32,32,CLOSEST,12,12,256,KEEP_SIGN,12) == 512);
    }

    printf("  Function \"code_image_coding_tmps_length\".\n");

    {
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,10,1) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,10,2) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,10,3) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,10,9) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,10,14) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,10,28) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION_ORDER,100,10,1) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION_ORDER,100,10,2) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION_ORDER,100,10,3) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION_ORDER,100,10,9) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION_ORDER,100,10,14) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION_ORDER,100,10,28) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,MATCHING_PURSUIT,100,10,1) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,MATCHING_PURSUIT,100,10,2) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,MATCHING_PURSUIT,100,10,3) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,MATCHING_PURSUIT,100,10,9) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,MATCHING_PURSUIT,100,10,14) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,MATCHING_PURSUIT,100,10,28) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,20,1) == 9*9*sizeof(double) + 28*28*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,20,2) == 9*9*sizeof(double) + 28*28*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,20,3) == 9*9*sizeof(double) + 27*27*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,20,9) == 9*9*sizeof(double) + 27*27*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,20,14) == 9*9*sizeof(double) + 28*28*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,100,20,28) == 9*9*sizeof(double) + 28*28*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,200,10,1) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,200,10,2) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,200,10,3) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,200,10,9) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,200,10,14) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,IDENTITY,0,0,CORRELATION,200,10,28) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(32,32,15,15,CLOSEST,15,15,CORRELATION,100,10,1) == 15*15*sizeof(double) + 15*15*sizeof(double) + 15*15*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(32,32,15,15,CLOSEST,15,15,CORRELATION,100,10,2) == 15*15*sizeof(double) + 15*15*sizeof(double) + 14*14*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(32,32,15,15,CLOSEST,15,15,CORRELATION,100,10,3) == 15*15*sizeof(double) + 15*15*sizeof(double) + 15*15*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(32,32,15,15,CLOSEST,15,15,CORRELATION,100,10,9) == 15*15*sizeof(double) + 15*15*sizeof(double) + 9*9*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(32,32,15,15,CLOSEST,15,15,CORRELATION,100,10,14) == 15*15*sizeof(double) + 15*15*sizeof(double) + 14*14*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 14*14*sizeof(size_t));
    }

    printf("  Function \"code_image\".\n");

    printf("    With Identity resize, Corr coding, Linear nonlinearity, None polarity split and Subsample reduce.\n");

    {
	size_t                    o_coeffs_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum resize_type          resize_type = IDENTITY;
        size_t                    new_row_count = 0;
        size_t                    new_col_count = 0;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = {0.5774,0.0000,0.0000,0.5774,0.0000,0.0000,
					    0.5774,0.0000,0.0000,0.0000,0.5774,0.0000,
					    0.5774,0.0000,0.0000,0.0000,0.0000,0.5774,
					    0.0000,0.5774,0.0000,0.5774,0.0000,0.0000,
					    0.0000,0.5774,0.0000,0.0000,0.5774,0.0000,
					    0.0000,0.5774,0.0000,0.0000,0.0000,0.5774,
					    0.0000,0.0000,0.5774,0.5774,0.0000,0.0000,
					    0.0000,0.0000,0.5774,0.0000,0.5774,0.0000,
					    0.0000,0.0000,0.5774,0.0000,0.0000,0.5774};
        double                    dict_transp[] = {0.5774,0.5774,0.5774,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
						   0.0000,0.0000,0.0000,0.5774,0.5774,0.5774,0.0000,0.0000,0.0000,
						   0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.5774,0.5774,0.5774,
						   0.5774,0.0000,0.0000,0.5774,0.0000,0.0000,0.5774,0.0000,0.0000,
						   0.0000,0.5774,0.0000,0.0000,0.5774,0.0000,0.0000,0.5774,0.0000,
						   0.0000,0.0000,0.5774,0.0000,0.0000,0.5774,0.0000,0.0000,0.5774};
        double                    dict_x_dict_transp[] = {1.0000,0.0000,0.0000,0.3333,0.3333,0.3333,
							  0.0000,1.0000,0.0000,0.3333,0.3333,0.3333,
							  0.0000,0.0000,1.0000,0.3333,0.3333,0.3333,
							  0.3333,0.3333,0.3333,1.0000,0.0000,0.0000,
							  0.3333,0.3333,0.3333,0.0000,1.0000,0.0000,
							  0.3333,0.3333,0.3333,0.0000,0.0000,1.0000};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = SUBSAMPLE;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,resize_type,new_row_count,new_col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,resize_type,new_row_count,new_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeffs_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   resize_type,new_row_count,new_col_count,coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	printf("      Not yet tested!!!\n");

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);                 
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Linear nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, Corr coding, Logistic nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Linear nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, CorrOrder coding, Logistic nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Linear nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Identity resize, MP coding, Logistic nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Linear nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, Corr coding, Logistic nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Linear nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, CorrOrder coding, Logistic nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Linear nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, None polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("    With Closest resize, MP coding, Logistic nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	printf("      Not yet tested!!!\n");
    }

    printf("Testing \"task_control\".\n");

    printf("  Function \"run_workers\".\n");

    {
        struct task_info  task_info[4];
        struct timeval    start_time;
        struct timeval    end_time;
        time_t            time_used_sec;

        task_info[0].o_result = -1;
        task_info[0].value = 10;
        task_info[1].o_result = -1;
        task_info[1].value = 20;
        task_info[2].o_result = -1;
        task_info[2].value = 30;
        task_info[3].o_result = -1;
        task_info[3].value = 40;

        gettimeofday(&start_time,NULL);

        run_workers(2,(task_fn_t)_do_work,4,task_info,sizeof(struct task_info));

        gettimeofday(&end_time,NULL);

        time_used_sec = end_time.tv_sec - start_time.tv_sec;

        assert(task_info[0].o_result == 20);
        assert(task_info[0].value == 10);
        assert(task_info[1].o_result == 40);
        assert(task_info[1].value == 20);
        assert(task_info[2].o_result == 60);
        assert(task_info[2].value == 30);
        assert(task_info[3].o_result == 80);
        assert(task_info[3].value == 40);
        assert(time_used_sec  >= 6);
        assert(time_used_sec <= 9);
    }

    printf("  Function \"run_workers_x\".\n");

    {
        struct global_info_x  global_info;
	struct global_vars_x  global_vars;
        struct task_info_x    task_info[4];
        struct timeval        start_time;
        struct timeval        end_time;
        time_t                time_used_sec;

        global_info.alpha = 10;
        global_info.beta = 15;
	global_vars.counter = 0;
	pthread_mutex_init(&global_vars.counter_lock,NULL);
        task_info[0].o_result = -1;
        task_info[0].value = 10;
        task_info[1].o_result = -1;
        task_info[1].value = 20;
        task_info[2].o_result = -1;
        task_info[2].value = 30;
        task_info[3].o_result = -1;
        task_info[3].value = 40;

        gettimeofday(&start_time,NULL);

        run_workers_x(&global_info,&global_vars,4,sizeof(struct task_info_x),task_info,(task_fn_x_t)_do_work_x,2);

        gettimeofday(&end_time,NULL);

        time_used_sec = end_time.tv_sec - start_time.tv_sec;

	pthread_mutex_destroy(&global_vars.counter_lock);

        assert(task_info[0].o_result == 115);
        assert(task_info[0].value == 10);
        assert(task_info[1].o_result == 215);
        assert(task_info[1].value == 20);
        assert(task_info[2].o_result == 315);
        assert(task_info[2].value == 30);
        assert(task_info[3].o_result == 415);
        assert(task_info[3].value == 40);
	assert(global_vars.counter == 4);
        assert(time_used_sec >= 6);
        assert(time_used_sec <= 9);
    }

    {
        struct global_info_x  global_info;
	struct global_vars_x  global_vars;
        struct task_info_x    task_info[5];
        struct timeval        start_time;
        struct timeval        end_time;
        time_t                time_used_sec;

        global_info.alpha = 10;
        global_info.beta = 15;
	global_vars.counter = 0;
	pthread_mutex_init(&global_vars.counter_lock,NULL);
        task_info[0].o_result = -1;
        task_info[0].value = 10;
        task_info[1].o_result = -1;
        task_info[1].value = 20;
        task_info[2].o_result = -1;
        task_info[2].value = 30;
        task_info[3].o_result = -1;
        task_info[3].value = 40;
        task_info[4].o_result = -1;
        task_info[4].value = 50;

        gettimeofday(&start_time,NULL);

        run_workers_x(&global_info,&global_vars,5,sizeof(struct task_info_x),task_info,(task_fn_x_t)_do_work_x,2);

        gettimeofday(&end_time,NULL);

        time_used_sec = end_time.tv_sec - start_time.tv_sec;

	pthread_mutex_destroy(&global_vars.counter_lock);

        assert(task_info[0].o_result == 115);
        assert(task_info[0].value == 10);
        assert(task_info[1].o_result == 215);
        assert(task_info[1].value == 20);
        assert(task_info[2].o_result == 315);
        assert(task_info[2].value == 30);
        assert(task_info[3].o_result == 415);
        assert(task_info[3].value == 40);
        assert(task_info[4].o_result == 515);
        assert(task_info[4].value == 50);
	assert(global_vars.counter == 5);
        assert(time_used_sec >= 9);
        assert(time_used_sec <= 12);
    }

    return EXIT_SUCCESS;
}
