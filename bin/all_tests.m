%% Setup tests.

DISPLAY = false;

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
datasets.record.test(DISPLAY);
datasets.image.test(DISPLAY);

%% Tests for "transform" and derived classes.

transforms.dc_offset.test(DISPLAY);
transforms.mean_substract.test(DISPLAY);
transforms.pca.test(DISPLAY);
transforms.pca_whitening.test(DISPLAY);
transforms.zca.test(DISPLAY);
transforms.image.resize.test(DISPLAY);
transforms.image.window.test(DISPLAY);
transforms.image.patch_extract.test(DISPLAY);
transforms.image.remove_pinknoise.test(DISPLAY);
transforms.image.random_corr.test(DISPLAY);
transforms.sparse.gdmp.test(DISPLAY);
transforms.sparse.sgdmp.test(DISPLAY);

%% Tests for "classifier" and derived classes.

classifiers.cmeans.test(DISPLAY);
classifiers.knn.test(DISPLAY);
classifiers.logistic_regression.test(DISPLAY);
classifiers.svm.test(DISPLAY);
classifiers.one_vs_one.test(DISPLAY);
