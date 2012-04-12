MNIST_FULL_IMAGES_PATH = '../data/mnist/train-images-idx3-ubyte';
MNIST_FULL_LABELS_PATH = '../data/mnist/train-labels-idx1-ubyte';
MNIST_TEST_IMAGES_PATH = '../data/mnist/t10k-images-idx3-ubyte';
MNIST_TEST_LABELS_PATH = '../data/mnist/t10k-labels-idx1-ubyte';
NEW_DIGIT_ROW_COUNT = 14;
NEW_DIGIT_COL_COUNT = 14;
TRAIN_CROSSVAL_RATIO = 0.2;
DETAILS_LOGFILE_PATH = sprintf('../explogs/mnist_arch_pca_%s.log',datestr(now,'yyyy-mm-dd_HH-MM'));
RESULTS_EMAIL_ADDRS = {'coman@inb.uni-luebeck.de'};
ERROR_EMAIL_ADDRS = {'coman@inb.uni-luebeck.de'};
RESULTS_ERROR_SENDER = 'coman@inb.uni-luebeck.de';

svm_classifier = @(s,kt,kp,l)classifiers.one_vs_one(s,@classifiers.svm,{kt kp},l);

params_desc.kept_energy = [0.95 0.96 0.97 0.98 0.99];
params_desc.classifier_ctor_fn = {@classifiers.cmeans svm_classifier};
params_desc.kernel_type = params.condition('classifier_ctor_fn',@classifiers.cmeans,{},...
                                                                svm_classifier,{'linear' 'rbf'});
params_desc.kernel_param = params.condition('kernel_type',{},{},...
                                                         'linear',0,...
                                                         'rbf', 2:0.25:6);
                                                     
experiments.small_images('MNIST: PCA on linearized images with CMeans and SVM classifiers',...
                         MNIST_FULL_IMAGES_PATH,MNIST_FULL_LABELS_PATH,MNIST_TEST_IMAGES_PATH,MNIST_TEST_LABELS_PATH,...
                         NEW_DIGIT_ROW_COUNT,NEW_DIGIT_COL_COUNT,...
                         params_desc,{},TRAIN_CROSSVAL_RATIO,...
                         @(s,p,l)architectures.arch_pca(s,p.kept_energy,p.classifier_ctor_fn,utils.cell_cull({p.kernel_type,p.kernel_param}),l),...
                         DETAILS_LOGFILE_PATH,RESULTS_EMAIL_ADDRS,ERROR_EMAIL_ADDRS,RESULTS_ERROR_SENDER);
