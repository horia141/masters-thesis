classdef svm_classifier < binary_classifier
    properties (GetAccess=public,SetAccess=immutable)
        kernel_type;
        kernel_param;
        svm;
    end
    
    methods (Access=public)
        function [obj] = svm_classifier(train_samples,kernel_type,kernel_param)
            assert(tc.scalar(train_samples) && tc.samples_set(train_samples));
            assert(train_samples.samples_count > 0);
            assert(length(unique(train_samples.labels_idx)) == 2);
            assert(tc.string(kernel_type) && (strcmp(kernel_type,'linear') || ...
                   strcmp(kernel_type,'rbf') || strcmp(kernel_type,'poly')));
            assert((strcmp(kernel_type,'linear') && ...
                     (~exist('kernel_param','var') || ...
                       (tc.scalar(kernel_param) && tc.number(kernel_param) && (kernel_param == 0)))) || ...
                   (strcmp(kernel_type,'rbf') && ...
                     (tc.scalar(kernel_param) && tc.number(kernel_param) && (kernel_param > 0))) || ...
                   (strcmp(kernel_type,'poly') && ...
                     (tc.scalar(kernel_param) && tc.natural(kernel_param) && (kernel_param > 0))));
                 
            if strcmp(kernel_type,'linear')
                svm_t = svmtrain(train_samples.samples,train_samples.labels_idx,'kernel_function','linear');
                kernel_param_t = 0;
            elseif strcmp(kernel_type,'rbf')
                svm_t = svmtrain(train_samples.samples,train_samples.labels_idx,'kernel_function','rbf','rbf_sigma',kernel_param);
                kernel_param_t = kernel_param;
            elseif strcmp(kernel_type,'poly')
                svm_t = svmtrain(train_samples.samples,train_samples.labels_idx,'kernel_function','poly','polyorder',kernel_param);
                kernel_param_t = kernel_param;
            else
                assert(false);
            end
            
            obj = obj@binary_classifier(train_samples);
            obj.kernel_type = kernel_type;
            obj.kernel_param = kernel_param_t;
            obj.svm = svm_t;
        end
    end
       
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = do_classify(obj,samples)
            labels_idx_hat = svmclassify(obj.svm,samples.samples);
            labels_confidence = ones(size(labels_idx_hat));
            labels_idx_hat2 = repmat(1:samples.classes_count,samples.samples_count,1);
            labels_confidence2 = zeros(samples.samples_count,samples.classes_count);

            for i = 1:samples.samples_count
                labels_idx_hat2(i,labels_idx_hat(i)) = 1;
                labels_idx_hat2(i,1) = labels_idx_hat(i);
                labels_confidence2(i,1) = 1;
            end
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "svm_classifier".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With linear kernel and param=0 (default).\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1)];
            
            s = samples_set({'1' '2'},A,c);
            cl = svm_classifier(s,'linear');
            
            assert(strcmp(cl.kernel_type,'linear'));
            assert(cl.kernel_param == 0);
            assert(strcmp(func2str(cl.svm.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.svm.KernelFunctionArgs));
            assert(tc.check(cl.svm.GroupNames == c));
            
            clearvars -except display;
            
            fprintf('    With linear kernel and param=0.\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1)];
            
            s = samples_set({'1' '2'},A,c);
            cl = svm_classifier(s,'linear',0);
            
            assert(strcmp(cl.kernel_type,'linear'));
            assert(cl.kernel_param == 0);
            assert(strcmp(func2str(cl.svm.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.svm.KernelFunctionArgs));
            assert(tc.check(cl.svm.GroupNames == c));
            
            clearvars -except display;
            
            fprintf('    With rbf kernel.\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1)];
            
            s = samples_set({'1' '2'},A,c);
            cl = svm_classifier(s,'rbf',0.7);
            
            assert(strcmp(cl.kernel_type,'rbf'));
            assert(cl.kernel_param == 0.7);
            assert(strcmp(func2str(cl.svm.KernelFunction),'rbf_kernel'));
            assert(cl.svm.KernelFunctionArgs{1} == 0.7);
            assert(tc.check(cl.svm.GroupNames == c));
            
            clearvars -except display;
            
            fprintf('    With poly kernel.\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1)];
            
            s = samples_set({'1' '2'},A,c);
            cl = svm_classifier(s,'poly',3);
            
            assert(strcmp(cl.kernel_type,'poly'));
            assert(cl.kernel_param == 3);
            assert(strcmp(func2str(cl.svm.KernelFunction),'poly_kernel'));
            assert(cl.svm.KernelFunctionArgs{1} == 3);
            assert(tc.check(cl.svm.GroupNames == c));
            
            clearvars -except display;
            
            fprintf('    With linear kernel and more than two classes possible.\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1' '2' '3' '4'},A,c);
            cl = svm_classifier(s,'linear');
            
            assert(strcmp(cl.kernel_type,'linear'));
            assert(cl.kernel_param == 0);
            assert(strcmp(func2str(cl.svm.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.svm.KernelFunctionArgs));
            assert(tc.check(cl.svm.GroupNames == c));
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);2*ones(100,1)];
            
            s = samples_set({'1' '2'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = svm_classifier(s_tr,'linear');
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(tc.check(labels_confidence == ones(40,1)));
            assert(tc.check(labels_idx_hat2 == [repmat([1 2],20,1);repmat([2 1],20,1)]));
            assert(tc.check(labels_confidence2 == [ones(40,1),zeros(40,1)]));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With mostly clearly separated data.\n');
            
            A_tr = [mvnrnd([3 1],[0.01 0; 0 0.01],80);
                    mvnrnd([1 3],[0.01 0; 0 0.01],80)];
            A_ts = [mvnrnd([3 1],[0.01 0; 0 0.01],19);
                    1 3;
                    mvnrnd([1 3],[0.01 0; 0 0.01],19)
                    3 1];
            c_tr = [1*ones(80,1);2*ones(80,1)];
            c_ts = [1*ones(20,1);2*ones(20,1)];
            
            s_tr = samples_set({'1' '2'},A_tr,c_tr);
            s_ts = samples_set({'1' '2'},A_ts,c_ts);
            cl = svm_classifier(s_tr,'linear');
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts);
            
            assert(tc.check(labels_idx_hat(1:19) == s_ts.labels_idx(1:19)));
            assert(tc.check(labels_idx_hat(21:39) == s_ts.labels_idx(21:39)));
            assert(labels_idx_hat(20) == 2);
            assert(labels_idx_hat(40) == 1);
            assert(tc.check(labels_confidence == ones(40,1)));
            assert(tc.check(labels_idx_hat2(1:19,:) == repmat([1 2],19,1)));
            assert(tc.check(labels_idx_hat2(21:39,:) == repmat([2 1],19,1)));
            assert(tc.check(labels_idx_hat2(20,:) == [2 1]));
            assert(tc.check(labels_idx_hat2(40,:) == [1 2]));
            assert(tc.check(labels_confidence2 == [ones(40,1),zeros(40,1)]));
            assert(score == 95);
            assert(tc.check(conf_matrix == [19 1; 1 19]));
            assert(tc.check(misclassified == [20 40]'));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data and using a linear kernel.\n');
            
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1)];
            
            s = samples_set({'1' '2'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = svm_classifier(s_tr,'linear');
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data and using an rbf kernel with sigma=0.5.\n');
            
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1)];
            
            s = samples_set({'1' '2'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = svm_classifier(s_tr,'rbf',0.5);
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data and using an rbf kernel with sigma=0.1.\n');
            
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1)];
            
            s = samples_set({'1' '2'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = svm_classifier(s_tr,'rbf',0.1);
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With clearly separated data and more than two classes possible.\n');
            
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1' '2' '3' '4'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = svm_classifier(s_tr,'linear');
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(tc.check(labels_confidence == ones(40,1)));
            assert(tc.check(labels_idx_hat2 == [repmat([1 2 3 4],20,1);repmat([3 2 1 4],20,1)]));
            assert(tc.check(labels_confidence2 == [ones(40,1),zeros(40,3)]));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2' '3' '4'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end