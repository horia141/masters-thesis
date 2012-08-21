%% Setup tests.

DISPLAY = false;

tic;

if DISPLAY
    test_figure = figure();
else
    test_figure = -1;
end

%% Tests for basic classes.

check.test(test_figure);
utils.common.test(test_figure);
utils.display.test(test_figure);
utils.params.test(test_figure);
dataset.test(test_figure);
classifier_info.test(test_figure);
regressor_info.test(test_figure);

%% Tests for "transform" and derived classes.

transforms.record.dc_offset.test(test_figure);
transforms.record.mean_substract.test(test_figure);
transforms.record.normalize.test(test_figure);
transforms.record.standardize.test(test_figure);
transforms.record.pca.test(test_figure);
transforms.record.zca.test(test_figure);
transforms.record.dictionary.test(test_figure);
transforms.record.dictionary.given.dct.test(test_figure);
transforms.record.dictionary.random.filters.test(test_figure);
transforms.record.dictionary.random.instances.test(test_figure);
transforms.record.dictionary.learn.grad.test(test_figure);
transforms.record.dictionary.learn.grad_st.test(test_figure);
transforms.record.dictionary.learn.neural_gas.test(test_figure);
transforms.image.resize.test(test_figure);
transforms.image.patch_extract.test(test_figure);
transforms.image.digit.deform.test(test_figure);
transforms.image.recoder.test(test_figure);

%% Tests for "classifier" and derived classes.

classifiers.cmeans.test(test_figure);
classifiers.knn.test(test_figure);
classifiers.linear.logistic.test(test_figure);
classifiers.linear.svm.test(test_figure);

%% Tests for external stuff.

system('./+xtern/test');

%% Print timing results.

fprintf('Total test time: %.0fs\n',toc());
