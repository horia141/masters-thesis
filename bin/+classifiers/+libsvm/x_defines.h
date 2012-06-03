#ifndef _X_DEFINES_H
#define _X_DEFINES_H

static const char*
SVM_TYPE_TO_STRING[] = {
    /* 0: C_SVC */  "C-SVC"
};

static const double
EPS_DEFAULT[] = {
    /* 0: C_SVC */ 0.001
};

static const char*
KERNEL_CODE_TO_STRING[] = {
    /* 0: LINEAR */   "Linear",
    /* 1: POLY */     "Polynomial",
    /* 2: RBF */      "Gaussian",
    /* 3: SIGMOID */  "Logistic"
};

static const char*
KERNEL_PARAM1_TO_STRING[] = {
    /* 0: LINEAR */   "Unused",
    /* 1: POLY */     "Degree",
    /* 2: RBF */      "Gamma",
    /* 3: SIGMOID */  "Gamma"
};

static const char*
KERNEL_PARAM2_TO_STRING[] = {
    /* 0: LINEAR */   "Unused",
    /* 1: POLY */     "Coeff0",
    /* 2: RBF */      "Unused",
    /* 3: SIGMOID */  "Coeff0"
};

static double
compute_cache_size(
    int  sample_count) {
    return fmax(10,(((double)sample_count * (double)sample_count * sizeof(double)) / (double)(1024 * 1024)));
}

#endif
