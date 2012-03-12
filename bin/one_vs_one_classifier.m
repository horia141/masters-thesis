classdef one_vs_one_classifier < classifier
    properties (GetAccess=public,SetAccess=immutable)
        classifiers;
        classifiers_count;
    end
    
    methods (Access=public)
        function [obj] = one_vs_one_classifier(train_samples,classifier_ctor_fn,params)
            assert(tc.scalar(train_samples) && tc.samples_set(train_samples));
            assert(tc.scalar(classifier_ctor_fn) && tc.function_h(classifier_ctor_fn));
            assert(tc.vector(params) && tc.cell(params) && ...
                    (tc.check(cellfun(@(x)tc.value(x),params)) || ...
                     (tc.check(cellfun(@(x)tc.vector(x) && tc.cell(x) && ...
                                           tc.check(cellfun(@(y)tc.value(y),x)),params)) && ...
                      length(params) == 0.5 * train_samples.classes_count * (train_samples.classes_count - 1))));
                  
            if tc.check(cellfun(@(x)tc.value(x),params))
                one_param_set = true;
            else
                one_param_set = false;
            end
                
            current_classifier = 1;
                
            for i = 1:train_samples.classes_count
                for j = (i + 1):train_samples.classes_count
                    if one_param_set
                        local_params = params;
                    else
                        local_params = params{current_classifier};
                    end
                    
                    local_samples = train_samples.subsamples(train_samples.labels_idx == i | train_samples.labels_idx == j);
                    classifiers_t(current_classifier) = classifier_ctor_fn(local_samples,local_params{:});
                    current_classifier = current_classifier + 1;
                end
            end
            
            obj = obj@classifier(train_samples);
            obj.classifiers = classifiers_t;
            obj.classifiers_count = 0.5 * train_samples.classes_count * (train_samples.classes_count - 1);
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = do_classify(obj,samples)
            partial_labels_idx = zeros(samples.samples_count,obj.classifiers_count);
            
            for i = 1:obj.classifiers_count
                partial_labels_idx(:,i) = obj.classifiers(i).classify(samples);
            end

            votes = hist(partial_labels_idx',samples.classes_count);
            [~,labels_idx_hat] = max(votes,[],1);

            labels_idx_hat = labels_idx_hat';
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
            fprintf('Testing "one_vs_one_classifier".\n');
            
            fprintf('  Testing proper construction.\n');
            
            fprintf('    With SVM and common parameters (linear kernel).\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            cl = one_vs_one_classifier(s,@(samples,kernel_type)svm_classifier(samples,kernel_type),{'linear'});
            
            assert(strcmp(cl.classifiers(1).kernel_type,'linear'));
            assert(cl.classifiers(1).kernel_param == 0);
            assert(strcmp(func2str(cl.classifiers(1).svm.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifiers(1).svm.KernelFunctionArgs));
            assert(tc.check(cl.classifiers(1).svm.GroupNames == c(c == 1 | c == 2)));
            assert(strcmp(cl.classifiers(2).kernel_type,'linear'));
            assert(cl.classifiers(2).kernel_param == 0);
            assert(strcmp(func2str(cl.classifiers(2).svm.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifiers(2).svm.KernelFunctionArgs));
            assert(tc.check(cl.classifiers(2).svm.GroupNames == c(c == 1 | c == 3)));
            assert(strcmp(cl.classifiers(3).kernel_type,'linear'));
            assert(cl.classifiers(3).kernel_param == 0);
            assert(strcmp(func2str(cl.classifiers(3).svm.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifiers(3).svm.KernelFunctionArgs));
            assert(tc.check(cl.classifiers(3).svm.GroupNames == c(c == 2 | c == 3)));
            assert(cl.classifiers_count == 3);
            
            clearvars -except display;
            
            fprintf('    With SVM and common parameters (rbf kernel with sigma=0.7).\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            cl = one_vs_one_classifier(s,@(samples,kernel_type,kernel_param)svm_classifier(samples,kernel_type,kernel_param),{'rbf' 0.7});
            
            assert(strcmp(cl.classifiers(1).kernel_type,'rbf'));
            assert(cl.classifiers(1).kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifiers(1).svm.KernelFunction),'rbf_kernel'));
            assert(cl.classifiers(1).svm.KernelFunctionArgs{1} == 0.7);
            assert(tc.check(cl.classifiers(1).svm.GroupNames == c(c == 1 | c == 2)));
            assert(strcmp(cl.classifiers(2).kernel_type,'rbf'));
            assert(cl.classifiers(2).kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifiers(2).svm.KernelFunction),'rbf_kernel'));
            assert(cl.classifiers(2).svm.KernelFunctionArgs{1} == 0.7);
            assert(tc.check(cl.classifiers(2).svm.GroupNames == c(c == 1 | c == 3)));
            assert(strcmp(cl.classifiers(3).kernel_type,'rbf'));
            assert(cl.classifiers(3).kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifiers(3).svm.KernelFunction),'rbf_kernel'));
            assert(cl.classifiers(3).svm.KernelFunctionArgs{1} == 0.7);
            assert(tc.check(cl.classifiers(3).svm.GroupNames == c(c == 2 | c == 3)));
            assert(cl.classifiers_count == 3);
            
            clearvars -except display;
            
            fprintf('    With SVM and custom parameters.\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            cl = one_vs_one_classifier(s,@(samples,kernel_type,kernel_param)svm_classifier(samples,kernel_type,kernel_param),...
                                         {{'linear' 0} {'rbf' 0.7} {'poly' 3}});
            
            assert(strcmp(cl.classifiers(1).kernel_type,'linear'));
            assert(cl.classifiers(1).kernel_param == 0);
            assert(strcmp(func2str(cl.classifiers(1).svm.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifiers(1).svm.KernelFunctionArgs));
            assert(tc.check(cl.classifiers(1).svm.GroupNames == c(c == 1 | c == 2)));
            assert(strcmp(cl.classifiers(2).kernel_type,'rbf'));
            assert(cl.classifiers(2).kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifiers(2).svm.KernelFunction),'rbf_kernel'));
            assert(cl.classifiers(2).svm.KernelFunctionArgs{1} == 0.7);
            assert(tc.check(cl.classifiers(2).svm.GroupNames == c(c == 1 | c == 3)));
            assert(strcmp(cl.classifiers(3).kernel_type,'poly'));
            assert(cl.classifiers(3).kernel_param == 3);
            assert(strcmp(func2str(cl.classifiers(3).svm.KernelFunction),'poly_kernel'));
            assert(cl.classifiers(3).svm.KernelFunctionArgs{1} == 3);
            assert(tc.check(cl.classifiers(3).svm.GroupNames == c(c == 2 | c == 3)));
            assert(cl.classifiers_count == 3);

            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With SVMs with linear kernels on clearly separated data.\n');
            
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([3 3],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = one_vs_one_classifier(s_tr,@(samples,kernel_type)svm_classifier(samples,kernel_type),{'linear'});
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(tc.check(labels_confidence == ones(60,1)));
            assert(tc.check(labels_idx_hat2 == [repmat([1 2 3],20,1);repmat([2 1 3],20,1);repmat([3 2 1],20,1)]));
            assert(tc.check(labels_confidence2 == [ones(60,1),zeros(60,2)]));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With SVMs with linear kernels on mostly clearly separated data.\n');
            
            A_tr = [mvnrnd([3 1],[0.01 0; 0 0.01],80);
                    mvnrnd([3 3],[0.01 0; 0 0.01],80);
                    mvnrnd([1 3],[0.01 0; 0 0.01],80)];
            A_ts = [mvnrnd([3 1],[0.01 0; 0 0.01],18);
                    3 3;
                    1 3;
                    mvnrnd([3 3],[0.01 0; 0 0.01],18);
                    3 1;
                    1 3;
                    mvnrnd([1 3],[0.01 0; 0 0.01],18)
                    3 1;
                    3 3];
            c_tr = [1*ones(80,1);2*ones(80,1);3*ones(80,1)];
            c_ts = [1*ones(20,1);2*ones(20,1);3*ones(20,1)];
            
            s_tr = samples_set({'1' '2' '3'},A_tr,c_tr);
            s_ts = samples_set({'1' '2' '3'},A_ts,c_ts);
            cl = one_vs_one_classifier(s_tr,@(samples,kernel_type)svm_classifier(samples,kernel_type),{'linear'});
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts);
            
            assert(tc.check(labels_idx_hat(1:18) == s_ts.labels_idx(1:18)));
            assert(tc.check(labels_idx_hat(21:38) == s_ts.labels_idx(21:38)));
            assert(tc.check(labels_idx_hat(41:58) == s_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(tc.check(labels_confidence == ones(60,1)));
            assert(tc.check(labels_idx_hat2(1:18,:) == repmat([1 2 3],18,1)));
            assert(tc.check(labels_idx_hat2(21:38,:) == repmat([2 1 3],18,1)));
            assert(tc.check(labels_idx_hat2(41:58,:) == repmat([3 2 1],18,1)));
            assert(tc.check(labels_idx_hat2(19,:) == [2 1 3]));
            assert(tc.check(labels_idx_hat2(20,:) == [3 2 1]));
            assert(tc.check(labels_idx_hat2(39,:) == [1 2 3]));
            assert(tc.check(labels_idx_hat2(40,:) == [3 2 1]));
            assert(tc.check(labels_idx_hat2(59,:) == [1 2 3]));
            assert(tc.check(labels_idx_hat2(60,:) == [2 1 3]));
            assert(tc.check(labels_confidence2 == [ones(60,1),zeros(60,2)]));
            assert(score == 90);
            assert(tc.check(conf_matrix == [18 1 1; 1 18 1; 1 1 18]));
            assert(tc.check(misclassified == [19 20 39 40 59 60]'));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With SVMs with linear kernels on not so clearly separated data.\n');
            
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([3 3],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = one_vs_one_classifier(s_tr,@(samples,kernel_type)svm_classifier(samples,kernel_type),{'linear'});
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With SVMs with rbf kernel and sigma=0.5 on not so clearly separated data.\n');
            
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([3 3],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = one_vs_one_classifier(s_tr,@(samples,kernel_type,kernel_param)svm_classifier(samples,kernel_type,kernel_param),{'rbf' 0.5});
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With SVMs with rbf kernel and sigma=0.1 on not so clearly separated data.\n');
            
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([3 3],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = one_vs_one_classifier(s_tr,@(samples,kernel_type,kernel_param)svm_classifier(samples,kernel_type,kernel_param),{'rbf' 0.1});
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
