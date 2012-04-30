%% Setup tests.

DISPLAY = true;

tic;

%% Tests for basic classes.

tc.test(DISPLAY);
utils.test(DISPLAY);
utilstest.test(DISPLAY);
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
% regression_info.test(DISPLAY);

%% Tests for "transform" and derived classes.

transforms.record.dc_offset.test(DISPLAY);
transforms.record.mean_substract.test(DISPLAY);
transforms.record.pca.test(DISPLAY);
transforms.record.pca_whitening.test(DISPLAY);
transforms.record.zca.test(DISPLAY);
transforms.image.resize.test(DISPLAY);
transforms.image.window.test(DISPLAY);
transforms.image.patch_extract.test(DISPLAY);
transforms.image.remove_pinknoise.test(DISPLAY);
transforms.image.random_corr.test(DISPLAY);
transforms.sparse.gdmp2.test(DISPLAY);

%% Tests for "classifier" and derived classes.

classifiers.cmeans.test(DISPLAY);
classifiers.knn.test(DISPLAY);
classifiers.logistic_regression.test(DISPLAY);
classifiers.svm.test(DISPLAY);
classifiers.one_vs_all.test(DISPLAY);
classifiers.one_vs_one.test(DISPLAY);

%% Print timing results.

fprintf('Total test time: %.0fs\n',toc());