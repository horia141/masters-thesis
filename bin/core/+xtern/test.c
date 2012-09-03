#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#include "gsl/gsl_rng.h"

#include "base_defines.h"
#include "latools.h"
#include "coding_methods.h"
#include "image_coder.h"
#include "task_control.h"

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
        double  o_coeffs[] =     {1,3,-2,4,9,-5,7,6,0,-8};
        size_t  o_coeffs_idx[] = {0,1, 2,3,4, 5,6,7,8, 9};
        size_t  count = 10;

        sort_by_abs_coeffs(o_coeffs,o_coeffs_idx,count);

	assert(o_coeffs[0] == 9);
	assert(o_coeffs[1] == -8);
	assert(o_coeffs[2] == 7);
	assert(o_coeffs[3] == 6);
	assert(o_coeffs[4] == -5);
	assert(o_coeffs[5] == 4);
	assert(o_coeffs[6] == 3);
	assert(o_coeffs[7] == -2);
	assert(o_coeffs[8] == 1);
	assert(o_coeffs[9] == 0);
	assert(o_coeffs_idx[0] == 4);
	assert(o_coeffs_idx[1] == 9);
	assert(o_coeffs_idx[2] == 6);
	assert(o_coeffs_idx[3] == 7);
	assert(o_coeffs_idx[4] == 5);
	assert(o_coeffs_idx[5] == 3);
	assert(o_coeffs_idx[6] == 1);
	assert(o_coeffs_idx[7] == 2);
	assert(o_coeffs_idx[8] == 0);
	assert(o_coeffs_idx[9] == 8);
    }

    {
        double  o_coeffs[] =     {1,3,-3,2,7,9,-2,8,3,-4, 6,-3, 2, 1, 4,-5, 0, 5, 6,-7, 8, 9};
        size_t  o_coeffs_idx[] = {0,1, 2,3,4,5, 6,7,8, 9,10,11,12,13,14,15,16,17,18,19,20,21};
        size_t  count = 22;

        sort_by_abs_coeffs(o_coeffs,o_coeffs_idx,count);

	assert(o_coeffs[0] == 9);
	assert(o_coeffs[1] == 9);
	assert(o_coeffs[2] == 8);
	assert(o_coeffs[3] == 8);
	assert(o_coeffs[4] == 7);
	assert(o_coeffs[5] == -7);
	assert(o_coeffs[6] == 6);
	assert(o_coeffs[7] == 6);
	assert(o_coeffs[8] == -5);
	assert(o_coeffs[9] == 5);
	assert(o_coeffs[10] == -4);
	assert(o_coeffs[11] == 4);
	assert(o_coeffs[12] == 3);
	assert(o_coeffs[13] == -3);
	assert(o_coeffs[14] == 3);
	assert(o_coeffs[15] == -3);
	assert(o_coeffs[16] == 2);
	assert(o_coeffs[17] == -2);
	assert(o_coeffs[18] == 2);
	assert(o_coeffs[19] == 1);
	assert(o_coeffs[20] == 1);
	assert(o_coeffs[21] == 0);
	assert(o_coeffs_idx[0] == 5);
	assert(o_coeffs_idx[1] == 21);
	assert(o_coeffs_idx[2] == 7);
	assert(o_coeffs_idx[3] == 20);
	assert(o_coeffs_idx[4] == 4);
	assert(o_coeffs_idx[5] == 19);
	assert(o_coeffs_idx[6] == 10);
	assert(o_coeffs_idx[7] == 18);
	assert(o_coeffs_idx[8] == 15);
	assert(o_coeffs_idx[9] == 17);
	assert(o_coeffs_idx[10] == 9);
	assert(o_coeffs_idx[11] == 14);
	assert(o_coeffs_idx[12] == 1);
	assert(o_coeffs_idx[13] == 2);
	assert(o_coeffs_idx[14] == 8);
	assert(o_coeffs_idx[15] == 11);
	assert(o_coeffs_idx[16] == 3);
	assert(o_coeffs_idx[17] == 6);
	assert(o_coeffs_idx[18] == 12);
	assert(o_coeffs_idx[19] == 0);
	assert(o_coeffs_idx[20] == 13);
	assert(o_coeffs_idx[21] == 16);
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

    printf("  Function \"correlation_coding_tmps_length\".\n");

    {
	assert(correlation_coding_tmps_length(2,3,2) == 3 * sizeof(double) + 3 * sizeof(size_t));
	assert(correlation_coding_tmps_length(1,3,2) == 3 * sizeof(double) + 3 * sizeof(size_t));
	assert(correlation_coding_tmps_length(2,3,3) == 3 * sizeof(double) + 3 * sizeof(size_t));
	assert(correlation_coding_tmps_length(2,5,2) == 5 * sizeof(double) + 5 * sizeof(size_t));
    }

    printf("  Function \"matching_pursuit_coding_tmps_length\".\n");

    {
	assert(matching_pursuit_coding_tmps_length(2,3,2) == 3 * sizeof(double));
	assert(matching_pursuit_coding_tmps_length(1,3,2) == 3 * sizeof(double));
	assert(matching_pursuit_coding_tmps_length(2,3,3) == 3 * sizeof(double));
	assert(matching_pursuit_coding_tmps_length(2,5,2) == 5 * sizeof(double));
    }

    printf("  Function \"orthogonal_matching_pursuit_coding_tmps_length\".\n");

    {
	assert(orthogonal_matching_pursuit_coding_tmps_length(2,3,2) == 3 * sizeof(double) + 2 * sizeof(double) + 2 * 2 * sizeof(double) + 2 * 2 * sizeof(double));
	assert(orthogonal_matching_pursuit_coding_tmps_length(1,3,2) == 3 * sizeof(double) + 1 * sizeof(double) + 1 * 2 * sizeof(double) + 2 * 2 * sizeof(double));
	assert(orthogonal_matching_pursuit_coding_tmps_length(2,3,3) == 3 * sizeof(double) + 2 * sizeof(double) + 2 * 3 * sizeof(double) + 3 * 3 * sizeof(double));
	assert(orthogonal_matching_pursuit_coding_tmps_length(2,5,2) == 5 * sizeof(double) + 2 * sizeof(double) + 2 * 2 * sizeof(double) + 2 * 2 * sizeof(double));
    }

    printf("  Function \"optimized_orthogonal_matching_pursuit_coding_tmps_length\".\n");

    {
	assert(optimized_orthogonal_matching_pursuit_coding_tmps_length(2,3,2) == 3 * sizeof(bool) + 2 * sizeof(double) + 2 * 3 * sizeof(double) + 2 * 2 * sizeof(double) + 2 * 2 * sizeof(double) + 3 * sizeof(double) + 2 * sizeof(double));
	assert(optimized_orthogonal_matching_pursuit_coding_tmps_length(1,3,2) == 3 * sizeof(bool) + 1 * sizeof(double) + 1 * 3 * sizeof(double) + 1 * 2 * sizeof(double) + 2 * 2 * sizeof(double) + 3 * sizeof(double) + 1 * sizeof(double));
	assert(optimized_orthogonal_matching_pursuit_coding_tmps_length(2,3,3) == 3 * sizeof(bool) + 2 * sizeof(double) + 2 * 3 * sizeof(double) + 2 * 3 * sizeof(double) + 3 * 3 * sizeof(double) + 3 * sizeof(double) + 2 * sizeof(double));
	assert(optimized_orthogonal_matching_pursuit_coding_tmps_length(2,5,2) == 5 * sizeof(bool) + 2 * sizeof(double) + 2 * 5 * sizeof(double) + 2 * 2 * sizeof(double) + 2 * 2 * sizeof(double) + 5 * sizeof(double) + 2 * sizeof(double));
    }

    printf("  Function \"sparse_net_coding_tmps_length\".\n");

    {
	assert(sparse_net_coding_tmps_length(2,3,2) == 3 * sizeof(double) + 3 * sizeof(size_t) + 2 * sizeof(double));
	assert(sparse_net_coding_tmps_length(1,3,2) == 3 * sizeof(double) + 3 * sizeof(size_t) + 1 * sizeof(double));
	assert(sparse_net_coding_tmps_length(2,3,3) == 3 * sizeof(double) + 3 * sizeof(size_t) + 2 * sizeof(double));
	assert(sparse_net_coding_tmps_length(2,5,2) == 5 * sizeof(double) + 5 * sizeof(size_t) + 2 * sizeof(double));
    }

    printf("  Function \"correlation\".\n");

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,1,0,-1,1};
        double   dict_transp[] = {1,0,0,-1,1,1};
        double   dict_x_dict_transp[] = {1,0,1,0,1,1,1,1,2};
        size_t   coeff_count = 2;
        double   observation[] = {4,-3};
        char*    coding_tmps;
	char*    curr_coding_tmps;
        double*  similarities;
        size_t*  similarities_idx;

        coding_tmps = malloc(correlation_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
        similarities = (double*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(double);
        similarities_idx = (size_t*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(size_t);

        correlation(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(o_coeffs[0] == 4);
        assert(o_coeffs[1] == 3);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(similarities[0] == 4);
        assert(similarities[1] == 3);
        assert(similarities[2] == 1);
        assert(similarities_idx[0] == 0);
        assert(similarities_idx[1] == 1);
        assert(similarities_idx[2] == 2);

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
        size_t   coeff_count = 2;
        double   observation[] = {4,3};
        char*    coding_tmps;
	char*    curr_coding_tmps;
        double*  similarities;
        size_t*  similarities_idx;

        coding_tmps = malloc(correlation_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
        similarities = (double*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(double);
        similarities_idx = (size_t*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(size_t);

        correlation(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(o_coeffs[0] == 7);
        assert(o_coeffs[1] == 4);
        assert(o_coeffs_idx[0] == 2);
        assert(o_coeffs_idx[1] == 0);
        assert(similarities[0] == 7);
        assert(similarities[1] == 4);
        assert(similarities[2] == -3);
        assert(similarities_idx[0] == 2);
        assert(similarities_idx[1] == 0);
        assert(similarities_idx[2] == 1);

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
        size_t   coeff_count = 2;
        double   observation[] = {-4,3};
        char*    coding_tmps;
	char*    curr_coding_tmps;
        double*  similarities;
        size_t*  similarities_idx;

        coding_tmps = malloc(correlation_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
        similarities = (double*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(double);
        similarities_idx = (size_t*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(size_t);

        correlation(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(o_coeffs[0] == -4);
        assert(o_coeffs[1] == -3);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(similarities[0] == -4);
        assert(similarities[1] == -3);
        assert(similarities[2] == -1);
        assert(similarities_idx[0] == 0);
        assert(similarities_idx[1] == 1);
        assert(similarities_idx[2] == 2);

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
        size_t   coeff_count = 2;
        double   observation[] = {4,-3};
	char*    coding_tmps;
	char*    curr_coding_tmps;
        double*  similarities;

	coding_tmps = malloc(matching_pursuit_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
	similarities = (double*)coding_tmps;
	curr_coding_tmps += word_count * sizeof(double);

        matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(fabs(o_coeffs[0] - 4) < 1e-4);
        assert(fabs(o_coeffs[1] - 3) < 1e-4);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(fabs(similarities[0] - 0) < 1e-4);
        assert(fabs(similarities[1] - 0) < 1e-4);
        assert(fabs(similarities[2] - 0) < 1e-4);

	free(coding_tmps);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeff_count = 2;
        double   observation[] = {4,3};
	char*    coding_tmps;
	char*    curr_coding_tmps;
        double*  similarities;

	coding_tmps = malloc(matching_pursuit_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
	similarities = (double*)coding_tmps;
	curr_coding_tmps += word_count * sizeof(double);

        matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(fabs(o_coeffs[0] - 4.9497) < 1e-4);
        assert(fabs(o_coeffs[1] - 0.5000) < 1e-4);
        assert(o_coeffs_idx[0] == 2);
        assert(o_coeffs_idx[1] == 0);
        assert(fabs(similarities[0] - 0) < 1e-4);
        assert(fabs(similarities[1] - 0.4999) < 1e-4);
        assert(fabs(similarities[2] - (-0.3535)) < 1e-4);

	free(coding_tmps);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeff_count = 2;
        double   observation[] = {-4,3};
	char*    coding_tmps;
	char*    curr_coding_tmps;
        double*  similarities;

	coding_tmps = malloc(matching_pursuit_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
	similarities = (double*)coding_tmps;
	curr_coding_tmps += word_count * sizeof(double);

        matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(fabs(o_coeffs[0] - (-4)) < 1e-4);
        assert(fabs(o_coeffs[1] - (-3)) < 1e-4);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(fabs(similarities[0] - 0) < 1e-4);
        assert(fabs(similarities[1] - 0) < 1e-4);
        assert(fabs(similarities[2] - 0) < 1e-4);

	free(coding_tmps);
    }

    printf("  Function \"orthogonal_matching_pursuit\".\n");

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeff_count = 2;
        double   observation[] = {4,-3};
    	char*    coding_tmps;
    	char*    curr_coding_tmps;
    	double*  similarities;
    	double*  residual;
    	double*  dict_transp_normalized;
    	double*  coeff_inversion_matrix;

    	coding_tmps = malloc(orthogonal_matching_pursuit_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
	similarities = (double*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(double);
	residual = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * sizeof(double);
	dict_transp_normalized = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * coeff_count * sizeof(double);
	coeff_inversion_matrix = (double*)curr_coding_tmps;
	curr_coding_tmps += coeff_count * coeff_count * sizeof(double);

        orthogonal_matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(fabs(o_coeffs[0] - 4) < 1e-4);
        assert(fabs(o_coeffs[1] - 3) < 1e-4);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(fabs(similarities[0] - 0) < 1e-4);
        assert(fabs(similarities[1] - 3) < 1e-4);
        assert(fabs(similarities[2] - (-2.1213)) < 1e-4);
	assert(fabs(residual[0] - 0) < 1e-4);
	assert(fabs(residual[1] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[0] - 1) < 1e-4);
	assert(fabs(dict_transp_normalized[1] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[2] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[3] - (-1)) < 1e-4);
	assert(fabs(coeff_inversion_matrix[0] - 1) < 1e-4);
	assert(fabs(coeff_inversion_matrix[2] - 0) < 1e-4);
	assert(fabs(coeff_inversion_matrix[3] - 1) < 1e-4);

	free(coding_tmps);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeff_count = 2;
        double   observation[] = {4,3};
    	char*    coding_tmps;
    	char*    curr_coding_tmps;
    	double*  similarities;
    	double*  residual;
    	double*  dict_transp_normalized;
    	double*  coeff_inversion_matrix;

    	coding_tmps = malloc(orthogonal_matching_pursuit_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
	similarities = (double*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(double);
	residual = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * sizeof(double);
	dict_transp_normalized = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * coeff_count * sizeof(double);
	coeff_inversion_matrix = (double*)curr_coding_tmps;
	curr_coding_tmps += coeff_count * coeff_count * sizeof(double);

        orthogonal_matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(fabs(o_coeffs[0] - 4.2427) < 1e-4);
        assert(fabs(o_coeffs[1] - 1) < 1e-4);
        assert(o_coeffs_idx[0] == 2);
        assert(o_coeffs_idx[1] == 0);
        assert(fabs(similarities[0] - 0.5001) < 1e-4);
        assert(fabs(similarities[1] - 0.4999) < 1e-4);
        assert(fabs(similarities[2] - 0.0001) < 1e-4);
	assert(fabs(residual[0] - 0) < 1e-4);
	assert(fabs(residual[1] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[0] - 0.7071) < 1e-4);
	assert(fabs(dict_transp_normalized[1] - 0.7071) < 1e-4);
	assert(fabs(dict_transp_normalized[2] - 0.7071) < 1e-4);
	assert(fabs(dict_transp_normalized[3] - (-0.7071)) < 1e-4);
	assert(fabs(coeff_inversion_matrix[0] - 1) < 1e-4);
	assert(fabs(coeff_inversion_matrix[2] - 0.7071) < 1e-4);
	assert(fabs(coeff_inversion_matrix[3] - 0.7071) < 1e-4);

	free(coding_tmps);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeff_count = 2;
        double   observation[] = {-4,3};
    	char*    coding_tmps;
    	char*    curr_coding_tmps;
    	double*  similarities;
    	double*  residual;
    	double*  dict_transp_normalized;
    	double*  coeff_inversion_matrix;

    	coding_tmps = malloc(orthogonal_matching_pursuit_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
	similarities = (double*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(double);
	residual = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * sizeof(double);
	dict_transp_normalized = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * coeff_count * sizeof(double);
	coeff_inversion_matrix = (double*)curr_coding_tmps;
	curr_coding_tmps += coeff_count * coeff_count * sizeof(double);

        orthogonal_matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(fabs(o_coeffs[0] - (-4)) < 1e-4);
        assert(fabs(o_coeffs[1] - (-3)) < 1e-4);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
        assert(fabs(similarities[0] - 0) < 1e-4);
        assert(fabs(similarities[1] - (-3)) < 1e-4);
        assert(fabs(similarities[2] - 2.1213) < 1e-4);
	assert(fabs(residual[0] - 0) < 1e-4);
	assert(fabs(residual[1] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[0] - 1) < 1e-4);
	assert(fabs(dict_transp_normalized[1] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[2] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[3] - (-1)) < 1e-4);
	assert(fabs(coeff_inversion_matrix[0] - 1) < 1e-4);
	assert(fabs(coeff_inversion_matrix[2] - 0) < 1e-4);
	assert(fabs(coeff_inversion_matrix[3] - 1) < 1e-4);

	free(coding_tmps);
    }

    printf("  Function \"optimized_orthogonal_matching_pursuit\".\n");

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeff_count = 2;
        double   observation[] = {4,-3};
    	char*    coding_tmps;
    	char*    curr_coding_tmps;
	bool*    used_column_mask;
	double*  residual;
	double*  dict_transp_tilde;
	double*  dict_transp_normalized;
	double*  coeff_inversion_matrix;

    	coding_tmps = malloc(optimized_orthogonal_matching_pursuit_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
	used_column_mask =  (bool*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(bool);
	residual = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * sizeof(double);
	dict_transp_tilde = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * word_count * sizeof(double);
	dict_transp_normalized = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * coeff_count * sizeof(double);
	coeff_inversion_matrix = (double*)curr_coding_tmps;
	curr_coding_tmps += coeff_count * coeff_count * sizeof(double);

        optimized_orthogonal_matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(fabs(o_coeffs[0] - 4) < 1e-4);
        assert(fabs(o_coeffs[1] - 3) < 1e-4);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
	assert(used_column_mask[0] == true);
	assert(used_column_mask[1] == true);
	assert(used_column_mask[2] == false);
	assert(fabs(residual[0] - 0) < 1e-4);
	assert(fabs(residual[1] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[0] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[1] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[2] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[3] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[4] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[5] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[0] - 1) < 1e-4);
	assert(fabs(dict_transp_normalized[1] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[2] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[3] - (-1)) < 1e-4);
	assert(fabs(coeff_inversion_matrix[0] - 1) < 1e-4);
	assert(fabs(coeff_inversion_matrix[2] - 0) < 1e-4);
	assert(fabs(coeff_inversion_matrix[3] - 1) < 1e-4);

	free(coding_tmps);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeff_count = 2;
        double   observation[] = {4,3};
    	char*    coding_tmps;
    	char*    curr_coding_tmps;
	bool*    used_column_mask;
	double*  residual;
	double*  dict_transp_tilde;
	double*  dict_transp_normalized;
	double*  coeff_inversion_matrix;

    	coding_tmps = malloc(optimized_orthogonal_matching_pursuit_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
	used_column_mask =  (bool*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(bool);
	residual = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * sizeof(double);
	dict_transp_tilde = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * word_count * sizeof(double);
	dict_transp_normalized = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * coeff_count * sizeof(double);
	coeff_inversion_matrix = (double*)curr_coding_tmps;
	curr_coding_tmps += coeff_count * coeff_count * sizeof(double);

        optimized_orthogonal_matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(fabs(o_coeffs[0] - 4.2427) < 1e-4);
        assert(fabs(o_coeffs[1] - 1) < 1e-4);
        assert(o_coeffs_idx[0] == 2);
        assert(o_coeffs_idx[1] == 0);
	assert(used_column_mask[0] == true);
	assert(used_column_mask[1] == false);
	assert(used_column_mask[2] == true);
	assert(fabs(residual[0] - 0) < 1e-4);
	assert(fabs(residual[1] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[0] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[1] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[2] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[3] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[4] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[5] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[0] - 0.7071) < 1e-4);
	assert(fabs(dict_transp_normalized[1] - 0.7071) < 1e-4);
	assert(fabs(dict_transp_normalized[2] - 0.7071) < 1e-4);
	assert(fabs(dict_transp_normalized[3] - (-0.7071)) < 1e-4);
	assert(fabs(coeff_inversion_matrix[0] - 1) < 1e-4);
	assert(fabs(coeff_inversion_matrix[2] - 0.7071) < 1e-4);
	assert(fabs(coeff_inversion_matrix[3] - 0.7071) < 1e-4);

	free(coding_tmps);
    }

    {
        double   o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t   o_coeffs_idx[] = {1000,1000};
        size_t   geometry = 2;
        size_t   word_count = 3;
        double   dict[] = {1,0,0.7071,0,-1,0.7071};
        double   dict_transp[] = {1,0,0,-1,0.7071,0.7071};
        double   dict_x_dict_transp[] = {1,0,0.7071,0,1,-0.7071,0.7071,-0.7071,1};
        size_t   coeff_count = 2;
        double   observation[] = {-4,3};
    	char*    coding_tmps;
    	char*    curr_coding_tmps;
	bool*    used_column_mask;
	double*  residual;
	double*  dict_transp_tilde;
	double*  dict_transp_normalized;
	double*  coeff_inversion_matrix;

    	coding_tmps = malloc(optimized_orthogonal_matching_pursuit_coding_tmps_length(geometry,word_count,coeff_count));
	curr_coding_tmps = coding_tmps;
	used_column_mask = (bool*)curr_coding_tmps;
	curr_coding_tmps += word_count * sizeof(bool);
	residual = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * sizeof(double);
	dict_transp_tilde = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * word_count * sizeof(double);
	dict_transp_normalized = (double*)curr_coding_tmps;
	curr_coding_tmps += geometry * coeff_count * sizeof(double);
	coeff_inversion_matrix = (double*)curr_coding_tmps;
	curr_coding_tmps += coeff_count * coeff_count * sizeof(double);

        optimized_orthogonal_matching_pursuit(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,observation,coding_tmps);

        assert(fabs(o_coeffs[0] - (-4)) < 1e-4);
        assert(fabs(o_coeffs[1] - (-3)) < 1e-4);
        assert(o_coeffs_idx[0] == 0);
        assert(o_coeffs_idx[1] == 1);
	assert(used_column_mask[0] == true);
	assert(used_column_mask[1] == true);
	assert(used_column_mask[2] == false);
	assert(fabs(residual[0] - 0) < 1e-4);
	assert(fabs(residual[1] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[0] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[1] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[2] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[3] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[4] - 0) < 1e-4);
	assert(fabs(dict_transp_tilde[5] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[0] - 1) < 1e-4);
	assert(fabs(dict_transp_normalized[1] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[2] - 0) < 1e-4);
	assert(fabs(dict_transp_normalized[3] - (-1)) < 1e-4);
	assert(fabs(coeff_inversion_matrix[0] - 1) < 1e-4);
	assert(fabs(coeff_inversion_matrix[2] - 0) < 1e-4);
	assert(fabs(coeff_inversion_matrix[3] - 1) < 1e-4);

	free(coding_tmps);
    }

    printf("  Function \"sparse_net\".\n");

    {
        double    o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t    o_coeffs_idx[] = {1000,1000};
        size_t    geometry = 2;
        size_t    word_count = 1;
        double    dict[] = {1,0};
        double    dict_transp[] = {1,0};
        double    dict_x_dict_transp[] = {1};
        size_t    coeff_count = 1;
	double    lambda_sigma_ratio = 0.1;
	gsl_rng*  rnd_generator;
	void*     param_table[2];
        double    observation[] = {4,-3};
        char*     coding_tmps;

        coding_tmps = malloc(sparse_net_coding_tmps_length(geometry,word_count,coeff_count));

	rnd_generator = gsl_rng_alloc(gsl_rng_mt19937);
	param_table[0] = &lambda_sigma_ratio;
	param_table[1] = rnd_generator;

        sparse_net(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,&param_table,observation,coding_tmps);

	gsl_rng_free(rnd_generator);
        free(coding_tmps);
    }

    {
        double    o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t    o_coeffs_idx[] = {1000,1000};
        size_t    geometry = 2;
        size_t    word_count = 3;
        double    dict[] = {1,0,1,0,-1,1};
        double    dict_transp[] = {1,0,0,-1,1,1};
        double    dict_x_dict_transp[] = {1,0,1,0,1,1,1,1,2};
        size_t    coeff_count = 2;
	double    lambda_sigma_ratio = 0.1;
	gsl_rng*  rnd_generator;
	void*     param_table[2];
        double    observation[] = {4,-3};
        char*     coding_tmps;

        coding_tmps = malloc(sparse_net_coding_tmps_length(geometry,word_count,coeff_count));

	rnd_generator = gsl_rng_alloc(gsl_rng_mt19937);
	param_table[0] = &lambda_sigma_ratio;
	param_table[1] = rnd_generator;

        sparse_net(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,&param_table,observation,coding_tmps);

	gsl_rng_free(rnd_generator);
        free(coding_tmps);
    }

    {
        double    o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t    o_coeffs_idx[] = {1000,1000};
        size_t    geometry = 2;
        size_t    word_count = 3;
        double    dict[] = {1,0,1,0,-1,1};
        double    dict_transp[] = {1,0,0,-1,1,1};
        double    dict_x_dict_transp[] = {1,0,1,0,1,1,1,1,2};
        size_t    coeff_count = 2;
	double    lambda_sigma_ratio = 0.1;
	gsl_rng*  rnd_generator;
	void*     param_table[2];
        double    observation[] = {4,3};
        char*     coding_tmps;

        coding_tmps = malloc(sparse_net_coding_tmps_length(geometry,word_count,coeff_count));

	rnd_generator = gsl_rng_alloc(gsl_rng_mt19937);
	param_table[0] = &lambda_sigma_ratio;
	param_table[1] = rnd_generator;

        sparse_net(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,&param_table,observation,coding_tmps);

	gsl_rng_free(rnd_generator);
        free(coding_tmps);
    }

    {
        double    o_coeffs[] = {HUGE_VAL,HUGE_VAL};
        size_t    o_coeffs_idx[] = {1000,1000};
        size_t    geometry = 2;
        size_t    word_count = 3;
        double    dict[] = {1,0,1,0,-1,1};
        double    dict_transp[] = {1,0,0,-1,1,1};
        double    dict_x_dict_transp[] = {1,0,1,0,1,1,1,1,2};
        size_t    coeff_count = 2;
	double    lambda_sigma_ratio = 0.1;
	gsl_rng*  rnd_generator;
	void*     param_table[2];
        double    observation[] = {-4,3};
        char*     coding_tmps;

        coding_tmps = malloc(sparse_net_coding_tmps_length(geometry,word_count,coeff_count));

	rnd_generator = gsl_rng_alloc(gsl_rng_mt19937);
	param_table[0] = &lambda_sigma_ratio;
	param_table[1] = rnd_generator;

        sparse_net(o_coeffs,o_coeffs_idx,geometry,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,&param_table,observation,coding_tmps);

	gsl_rng_free(rnd_generator);
        free(coding_tmps);
    }

    printf("Testing \"image_coder\".\n");

    printf("  Function \"code_image_new_geometry\".\n");

    {
	assert(code_image_new_geometry(28,28,100,NONE,1) == 78400);
	assert(code_image_new_geometry(28,28,100,NONE,2) == 19600);
	assert(code_image_new_geometry(28,28,100,NONE,3) == 8100);
	assert(code_image_new_geometry(28,28,100,NONE,4) == 4900);
	assert(code_image_new_geometry(28,28,100,NONE,5) == 2500);
	assert(code_image_new_geometry(28,28,100,NONE,6) == 1600);
	assert(code_image_new_geometry(28,28,100,NONE,7) == 1600);
	assert(code_image_new_geometry(28,28,100,NONE,8) == 900);
	assert(code_image_new_geometry(28,28,100,NONE,10) == 400);
	assert(code_image_new_geometry(28,28,100,NONE,11) == 400);
	assert(code_image_new_geometry(28,28,100,NONE,12) == 400);
	assert(code_image_new_geometry(28,28,100,NONE,13) == 400);
	assert(code_image_new_geometry(28,28,100,NONE,14) == 400);
	assert(code_image_new_geometry(28,28,100,NONE,15) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,16) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,17) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,18) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,19) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,20) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,21) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,22) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,23) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,24) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,25) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,26) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,27) == 100);
	assert(code_image_new_geometry(28,28,100,NONE,28) == 100);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,1) == 156800);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,2) == 39200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,3) == 16200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,4) == 9800);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,5) == 5000);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,6) == 3200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,7) == 3200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,8) == 1800);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,10) == 800);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,11) == 800);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,12) == 800);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,13) == 800);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,14) == 800);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,15) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,16) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,17) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,18) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,19) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,20) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,21) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,22) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,23) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,24) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,25) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,26) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,27) == 200);
	assert(code_image_new_geometry(28,28,100,NO_SIGN,28) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,1) == 156800);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,2) == 39200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,3) == 16200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,4) == 9800);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,5) == 5000);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,6) == 3200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,7) == 3200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,8) == 1800);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,10) == 800);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,11) == 800);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,12) == 800);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,13) == 800);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,14) == 800);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,15) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,16) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,17) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,18) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,19) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,20) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,21) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,22) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,23) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,24) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,25) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,26) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,27) == 200);
	assert(code_image_new_geometry(28,28,100,KEEP_SIGN,28) == 200);
    }

    printf("  Function \"code_image_coding_tmps_length\".\n");

    {
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,10,1) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,10,2) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,10,3) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,10,9) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,10,14) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,10,28) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,MATCHING_PURSUIT,100,10,1) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,MATCHING_PURSUIT,100,10,2) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,MATCHING_PURSUIT,100,10,3) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,MATCHING_PURSUIT,100,10,9) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,MATCHING_PURSUIT,100,10,14) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,MATCHING_PURSUIT,100,10,28) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,ORTHOGONAL_MATCHING_PURSUIT,100,10,1) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 9*9*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,ORTHOGONAL_MATCHING_PURSUIT,100,10,2) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 9*9*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,ORTHOGONAL_MATCHING_PURSUIT,100,10,3) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 9*9*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,ORTHOGONAL_MATCHING_PURSUIT,100,10,9) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 9*9*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,ORTHOGONAL_MATCHING_PURSUIT,100,10,14) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 9*9*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,ORTHOGONAL_MATCHING_PURSUIT,100,10,28) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 9*9*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,OPTIMIZED_ORTHOGONAL_MATCHING_PURSUIT,100,10,1) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(bool) + 9*9*sizeof(double) + 9*9*100*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 100*sizeof(double) + 9*9*sizeof(double) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,OPTIMIZED_ORTHOGONAL_MATCHING_PURSUIT,100,10,2) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(bool) + 9*9*sizeof(double) + 9*9*100*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 100*sizeof(double) + 9*9*sizeof(double) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,OPTIMIZED_ORTHOGONAL_MATCHING_PURSUIT,100,10,3) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(bool) + 9*9*sizeof(double) + 9*9*100*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 100*sizeof(double) + 9*9*sizeof(double) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,OPTIMIZED_ORTHOGONAL_MATCHING_PURSUIT,100,10,9) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(bool) + 9*9*sizeof(double) + 9*9*100*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 100*sizeof(double) + 9*9*sizeof(double) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,OPTIMIZED_ORTHOGONAL_MATCHING_PURSUIT,100,10,14) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(bool) + 9*9*sizeof(double) + 9*9*100*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 100*sizeof(double) + 9*9*sizeof(double) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,OPTIMIZED_ORTHOGONAL_MATCHING_PURSUIT,100,10,28) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(bool) + 9*9*sizeof(double) + 9*9*100*sizeof(double) + 9*9*10*sizeof(double) + 10*10*sizeof(double) + 100*sizeof(double) + 9*9*sizeof(double) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,SPARSE_NET,100,10,1) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 100*sizeof(size_t) + 9*9*sizeof(double) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,SPARSE_NET,100,10,2) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 100*sizeof(size_t) + 9*9*sizeof(double) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,SPARSE_NET,100,10,3) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 100*sizeof(size_t) + 9*9*sizeof(double) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,SPARSE_NET,100,10,9) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 100*sizeof(size_t) + 9*9*sizeof(double) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,SPARSE_NET,100,10,14) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 100*sizeof(size_t) + 9*9*sizeof(double) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,SPARSE_NET,100,10,28) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 100*sizeof(double) + 100*sizeof(size_t) + 9*9*sizeof(double) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,20,1) == 9*9*sizeof(double) + 28*28*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,20,2) == 9*9*sizeof(double) + 28*28*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,20,3) == 9*9*sizeof(double) + 27*27*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,20,9) == 9*9*sizeof(double) + 27*27*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,20,14) == 9*9*sizeof(double) + 28*28*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,100,20,28) == 9*9*sizeof(double) + 28*28*20*(sizeof(double) + sizeof(size_t)) + 100*(sizeof(double) + sizeof(size_t)) + 28*28*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,200,10,1) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 1*1*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,200,10,2) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 2*2*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,200,10,3) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 3*3*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,200,10,9) == 9*9*sizeof(double) + 27*27*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 9*9*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,200,10,14) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 14*14*sizeof(size_t));
	assert(code_image_coding_tmps_length(28,28,9,9,CORRELATION,200,10,28) == 9*9*sizeof(double) + 28*28*10*(sizeof(double) + sizeof(size_t)) + 200*(sizeof(double) + sizeof(size_t)) + 28*28*sizeof(size_t));
    }

    printf("Testing \"task_control\".\n");

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
