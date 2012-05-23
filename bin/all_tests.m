%% Setup tests.

DISPLAY = false;

tic;

%% Tests for basic classes.

tc.test(DISPLAY);
utils.test(DISPLAY);
params.test(DISPLAY);
logging.level.test(DISPLAY);
logging.handlers.zero.test(DISPLAY);
logging.handlers.stdout.test(DISPLAY);
logging.handlers.file.test(DISPLAY);
logging.handlers.sendmail.test(DISPLAY);
logging.handlers.testing.test(DISPLAY);
logging.logger.test(DISPLAY);

%% Tests for "dataset" and derived classes.

dataset.test(DISPLAY);
classification_info.test(DISPLAY);
regression_info.test(DISPLAY);

%% Tests for "transform" and derived classes.

transforms.record.dc_offset.test(DISPLAY);
transforms.record.mean_substract.test(DISPLAY);
transforms.record.normalize.test(DISPLAY);
transforms.record.standardize.test(DISPLAY);
transforms.record.pca.test(DISPLAY);
transforms.record.zca.test(DISPLAY);
transforms.record.dictionary.test(DISPLAY);
transforms.record.dictionary.random.filters.test(DISPLAY);
transforms.record.dictionary.random.instances.test(DISPLAY);
transforms.record.dictionary.given.fourier.test(DISPLAY);
transforms.record.dictionary.given.gabor.test(DISPLAY);
transforms.record.dictionary.learn.autoencoder.test(DISPLAY);
transforms.record.dictionary.learn.boltzmann.test(DISPLAY);
transforms.record.dictionary.learn.grad.test(DISPLAY);
transforms.record.dictionary.learn.grad_st.test(DISPLAY);
transforms.record.dictionary.learn.kmeans.test(DISPLAY);
transforms.record.dictionary.learn.neuralgas.test(DISPLAY);
transforms.image.resize.test(DISPLAY);
transforms.image.window.test(DISPLAY);
transforms.image.patch_extract.test(DISPLAY);
transforms.image.remove_pinknoise.test(DISPLAY);
transforms.image.digit.deform(DISPLAY);
transforms.image.digit.normalize_width(DISPLAY);
transforms.image.window_sparse_recoder.test(DISPLAY);

%% Tests for "classifier" and derived classes.

classifiers1.cmeans.test(DISPLAY);
classifiers1.knn.test(DISPLAY);
classifiers1.logistic.test(DISPLAY);
classifiers1.svm_linear.test(DISPLAY);
classifiers1.svm_c_kernel.test(DISPLAY);
classifiers1.svm_eps_kernel.test(DISPLAY);

%% Print timing results.

fprintf('Total test time: %.0fs\n',toc());