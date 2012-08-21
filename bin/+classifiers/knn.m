classdef knn < classifier
    properties (GetAccess=public,SetAccess=immutable)
        sample;
        labels_idx;
        k;
    end
    
    methods (Access=public)
        function [obj] = knn(train_sample,class_info,k)
            assert(check.dataset_record(train_sample));
            assert(check.scalar(class_info));
            assert(check.classifier_info(class_info));
            assert(check.scalar(k));
            assert(check.natural(k));
            assert(k >= 1);
            assert(k <= dataset.count(train_sample));
            assert(class_info.compatible(train_sample));
            
            input_geometry = dataset.geometry(train_sample);

            obj = obj@classifier(input_geometry,class_info.labels);
            obj.sample = train_sample;
            obj.labels_idx = class_info.labels_idx;
            obj.k = k;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample)
            N = dataset.count(sample);
            
            labels_idx_hat = knnclassify(sample',obj.sample',obj.labels_idx)';
            labels_confidence = zeros(obj.saved_labels_count,N);
            labels_confidence(sub2ind(size(labels_confidence),labels_idx_hat,1:N)) = 1;
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "classifiers.knn".\n');
            
            fprintf('  Proper construction.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.knn(s,ci,1);
            
            assert(check.same(cl.sample,s));
            assert(check.same(cl.labels_idx,ci.labels_idx));
            assert(cl.k == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            clearvars -except test_figure;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_3.test.mat');
            
            cl = classifiers.knn(s_tr,ci_tr,3);            
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.same(labels_confidence,[ones(20,1) zeros(20,1) zeros(20,1);zeros(20,1) ones(20,1) zeros(20,1);zeros(20,1) zeros(20,1) ones(20,1)]'));
            assert(score == 100);
            assert(check.same(conf_matrix,[20 0 0; 0 20 0; 0 0 20]));
            assert(check.empty(misclassified));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    With mostly clearly separated data.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_mostly_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_mostly_clear_data_3.test.mat');
            
            cl = classifiers.knn(s_tr,ci_tr,3);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);

            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(check.same(labels_confidence,[ones(18,1) zeros(18,1) zeros(18,1);0 1 0; 0 0 1;zeros(18,1) ones(18,1) zeros(18,1);1 0 0; 0 0 1;zeros(18,1) zeros(18,1) ones(18,1); 1 0 0; 0 1 0]'));
            assert(score == 90);
            assert(check.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(check.same(misclassified,[19 20 39 40 59 60]));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    Without clearly separated data (k=3).\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_unclear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_unclear_data_3.test.mat');
            
            cl = classifiers.knn(s_tr,ci_tr,3);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    Without clearly separated data (k=7).\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_unclear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_unclear_data_3.test.mat');
            
            cl = classifiers.knn(s_tr,ci_tr,7);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
        end
    end
end
