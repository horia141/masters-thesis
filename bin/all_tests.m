%% Setup tests.

DISPLAY = true;

%% Tests for basic classes.

tc.test(DISPLAY)
utils.test(DISPLAY)

%% Tests for "samples" and derived classes.

samples_set.test(DISPLAY)
gray_images_set.test(DISPLAY)

%% Tests for "transform" and derived classes.

dc_offset_transform.test(DISPLAY)
mean_substract_transform.test(DISPLAY)
pca_transform.test(DISPLAY)
pca_whitening_transform.test(DISPLAY)
zca_transform.test(DISPLAY)
patch_extract_transform.test(DISPLAY)
remove_pinknoise_transform.test(DISPLAY)

%% Tests for "classifier" and derived classes.

knn_classifier.test(DISPLAY)
svm_classifier.test(DISPLAY)
