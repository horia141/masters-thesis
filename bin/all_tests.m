%% Setup tests.

DISPLAY = false;

%% Tests for basic classes.

tc.test(DISPLAY);
utils.test(DISPLAY);
logging.level.test(DISPLAY);
logging.handlers.zero.test(DISPLAY);
logging.handlers.stdout.test(DISPLAY);
logging.handlers.file.test(DISPLAY);
logging.handlers.sendmail.test(DISPLAY);
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

knn_classifier.test(DISPLAY);
svm_classifier.test(DISPLAY);
cmeans_classifier.test(DISPLAY);
logistic_regression_classifier.test(DISPLAY);

one_vs_one_classifier.test(DISPLAY);

%% Tests for "architecture" and derived classes.

%baseline_1_architecture.test(DISPLAY);
%baseline_2_architecture.test(DISPLAY);
%random_cnn_1layer_architecture.test(DISPLAY);
