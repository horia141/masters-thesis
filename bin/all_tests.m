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

%% Tests for "samples" and derived classes.

samples_set.test(DISPLAY);
gray_images_set.test(DISPLAY);

%% Tests for "transform" and derived classes.

dc_offset_transform.test(DISPLAY);
mean_substract_transform.test(DISPLAY);
image_resize_transform.test(DISPLAY);
pca_transform.test(DISPLAY);
pca_whitening_transform.test(DISPLAY);
zca_transform.test(DISPLAY);
patch_extract_transform.test(DISPLAY);
remove_pinknoise_transform.test(DISPLAY);
%sparse_sgdmp_transform.test(DISPLAY);
%sparse_gdmp_transform.test(DISPLAY);

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
