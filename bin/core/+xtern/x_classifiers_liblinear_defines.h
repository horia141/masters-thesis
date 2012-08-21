#ifndef _X_CLASSIFIERS_LIBLINEAR_DEFINES_H
#define _X_CLASSIFIERS_LIBLINEAR_DEFINES_H

static const double
EPS_DEFAULT[] = {
    /* 0: L2R_LR */               0.01,
    /* 1: L2R_L2LOSS_SVC_DUAL */  0.1,
    /* 2: L2R_L2LOSS_SVC */       0.01,
    /* 3: L2R_L1LOSS_SVC_DUAL */  0.1,
    /* 4: unused */               0,
    /* 5: L1R_L2LOSS_SVC */       0.01,
    /* 6: L1R_LR */               0.01,
    /* 7: L2R_LR_DUAL */          0.1
};

#endif
