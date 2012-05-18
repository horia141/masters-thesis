MNIST_FULL_IMAGES_PATH = '../data/mnist/train-images-idx3-ubyte';
MNIST_FULL_LABELS_PATH = '../data/mnist/train-labels-idx1-ubyte';
MNIST_TEST_IMAGES_PATH = '../data/mnist/t10k-images-idx3-ubyte';
MNIST_TEST_LABELS_PATH = '../data/mnist/t10k-labels-idx1-ubyte';
TRAIN_CROSSVAL_RATIO = 0.2;
DETAILS_LOGFILE_PATH = sprintf('../explogs/mnist_random_corr_%s.log',datestr(now,'yyyy-mm-dd_HH-MM'));
RESULTS_EMAIL_ADDRS = {'coman@inb.uni-luebeck.de'};
ERROR_EMAIL_ADDRS = {'coman@inb.uni-luebeck.de'};
RESULTS_ERROR_SENDER = 'coman@inb.uni-luebeck.de';

params_desc.filters_count = 32:16:128;
params_desc.filter_row_count = [3 5 7 9];
params_desc.filter_col_count = params.depend('filter_row_count',@(v)v);
params_desc.reduce_function = {@transforms.image.random_corr.sqr @transforms.image.random_corr.max};
params_desc.reduce_spread = [2 3 4];
params_desc.classifier_ctor_fn = @(s,kt,kp,l)classifiers.one_vs_one(s,@classifiers.svm,{kt kp},l);
params_desc.kernel_type = {'linear' 'rbf'};
params_desc.kernel_param = params.condition('kernel_type','linear',0,...
                                                          'rbf', logspace(0,1,20));
                                                     
experiments.small_images('MNIST: Filtering using random bases with CMeans and SVM classifiers',...
                         MNIST_FULL_IMAGES_PATH,MNIST_FULL_LABELS_PATH,MNIST_TEST_IMAGES_PATH,MNIST_TEST_LABELS_PATH,...
                         28,28,...
                         params_desc,{@(p)mod(p.filter_row_count,2) == 1,@(p)mod(p.filter_col_count,2) == 1,...
                                      @(p)mod(28 - p.filter_row_count + 1,p.reduce_spread) == 0,...
                                      @(p)mod(28 - p.filter_col_count + 1,p.reduce_spread) == 0},TRAIN_CROSSVAL_RATIO,...
                         @(s,ci,p,l)architectures.arch_random_corr(s,ci,p.filters_count,p.filter_row_count,p.filter_col_count,p.reduce_function,p.reduce_spread,p.classifier_ctor_fn,utils.cell_cull({p.kernel_type,p.kernel_param}),l),...
                         DETAILS_LOGFILE_PATH,RESULTS_EMAIL_ADDRS,ERROR_EMAIL_ADDRS,RESULTS_ERROR_SENDER);
