#ifndef _X_DEFINES_H
#define _X_DEFINES_H

static const char*
METHOD_CODE_TO_STRING[] = {
    /* 0: L2R_LR */               "L2-regularized logistic regression in primal form",
    /* 1: L2R_L2LOSS_SVC_DUAL */  "L2-loss L2-regularized support vector classification in dual form",
    /* 2: L2R_L2LOSS_SVC */       "L2-loss L2-regularized support vector classification in primal form",
    /* 3: L2R_L1LOSS_SVC_DUAL */  "L1-loss L2-regularized support vector classification in dual form",
    /* 4: unused */               "Unused",
    /* 5: L1R_L2LOSS_SVC */       "L2-loss L1-regularized support vector classification in primal form",
    /* 6: L1R_LR */               "L1-regularized logistic regression in primal form",
    /* 7: L2R_LR_DUAL */          "L2-regularized logistic regression in dual form"
};

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
