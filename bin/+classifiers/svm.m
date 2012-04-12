classdef svm < classifiers.binary
    properties (GetAccess=public,SetAccess=immutable)
        svm_info;
        kernel_type;
        kernel_param;
    end
    
    methods (Access=public)
        function [obj] = svm(train_dataset,kernel_type,kernel_param,logger)
            assert(tc.scalar(train_dataset));
            assert(tc.dataset(train_dataset));
            assert(train_dataset.samples_count >= 1);
            assert(length(unique(train_dataset.labels_idx)) == 2);
            assert(tc.scalar(kernel_type));
            assert(tc.string(kernel_type));
            assert(tc.one_of(kernel_type,'linear','rbf','poly'));
            assert((strcmp(kernel_type,'linear') && ...
                     tc.scalar(kernel_param) && tc.number(kernel_param) && (kernel_param == 0)) || ...
                   (strcmp(kernel_type,'rbf') && ...
                     tc.scalar(kernel_param) && tc.number(kernel_param) && (kernel_param > 0)) || ...
                   (strcmp(kernel_type,'poly') && ...
                     tc.scalar(kernel_param) && tc.natural(kernel_param) && (kernel_param > 0)));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            logger.message('Computing separation surfaces.');
            
            try
                if strcmp(kernel_type,'linear')
                    svm_info_t = svmtrain(train_dataset.samples,train_dataset.labels_idx,'kernel_function','linear');
                    kernel_param_t = 0;
                elseif strcmp(kernel_type,'rbf')
                    svm_info_t = svmtrain(train_dataset.samples,train_dataset.labels_idx,'kernel_function','rbf','rbf_sigma',kernel_param);
                    kernel_param_t = kernel_param;
                elseif strcmp(kernel_type,'poly')
                    svm_info_t = svmtrain(train_dataset.samples,train_dataset.labels_idx,'kernel_function','poly','polyorder',kernel_param);
                    kernel_param_t = kernel_param;
                else
                    assert(false);
                end
            catch exp
                if strcmp(exp.identifier,'Bioinfo:svmtrain:NoConvergence')
                    throw(MException('master:NoConvergence',...
                             sprintf('Failed to converge with kernel "%s" and parameter "%f"',kernel_type,kernel_param)));
                else
                    rethrow(exp);
                end
            end
            
            actual_labels = unique(train_dataset.labels_idx);
            required_samples = [find(train_dataset.labels_idx == actual_labels(1),1) find(train_dataset.labels_idx == actual_labels(2),1)];
            
            obj = obj@classifiers.binary(train_dataset.subsamples(required_samples),logger);
            obj.svm_info = svm_info_t;
            obj.kernel_type = kernel_type;
            obj.kernel_param = kernel_param_t;
        end
    end
       
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = do_classify(obj,dataset_d,logger)
            logger.message('Computing dataset classes.');

            labels_idx_hat = svmclassify(obj.svm_info,dataset_d.samples);
            labels_confidence = ones(size(labels_idx_hat));
            labels_idx_hat2 = repmat(1:dataset_d.classes_count,dataset_d.samples_count,1);
            labels_confidence2 = zeros(dataset_d.samples_count,dataset_d.classes_count);

            for ii = 1:dataset_d.samples_count
                labels_idx_hat2(ii,labels_idx_hat(ii)) = 1;
                labels_idx_hat2(ii,1) = labels_idx_hat(ii);
                labels_confidence2(ii,1) = 1;
            end
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "classifiers.svm".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With linear kernel and param=0.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1)];            
            s = dataset({'1' '2'},A,c);
            
            cl = classifiers.svm(s,'linear',0,log);
            
            assert(length(cl.one_sample.classes) == 2);
            assert(strcmp(cl.one_sample.classes{1},'1'));
            assert(strcmp(cl.one_sample.classes{2},'2'));
            assert(cl.one_sample.classes_count == 2);
            assert(tc.check(cl.one_sample.samples == A(1,:)));
            assert(tc.check(cl.one_sample.labels_idx == c(1)));
            assert(cl.one_sample.samples_count == 1);
            assert(cl.one_sample.features_count == 2);
            assert(cl.one_sample.compatible(s));  
            assert(strcmp(cl.kernel_type,'linear'));
            assert(cl.kernel_param == 0);
            assert(strcmp(func2str(cl.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.svm_info.KernelFunctionArgs));
            assert(tc.check(cl.svm_info.GroupNames == c));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With rbf kernel.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1)];            
            s = dataset({'1' '2'},A,c);
            
            cl = classifiers.svm(s,'rbf',0.7,log);
            
            assert(length(cl.one_sample.classes) == 2);
            assert(strcmp(cl.one_sample.classes{1},'1'));
            assert(strcmp(cl.one_sample.classes{2},'2'));
            assert(cl.one_sample.classes_count == 2);
            assert(tc.check(cl.one_sample.samples == A(1,:)));
            assert(tc.check(cl.one_sample.labels_idx == c(1)));
            assert(cl.one_sample.samples_count == 1);
            assert(cl.one_sample.features_count == 2);
            assert(cl.one_sample.compatible(s));  
            assert(strcmp(cl.kernel_type,'rbf'));
            assert(cl.kernel_param == 0.7);
            assert(strcmp(func2str(cl.svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.check(cl.svm_info.GroupNames == c));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With poly kernel.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1)];            
            s = dataset({'1' '2'},A,c);
            
            cl = classifiers.svm(s,'poly',3,log);
            
            assert(length(cl.one_sample.classes) == 2);
            assert(strcmp(cl.one_sample.classes{1},'1'));
            assert(strcmp(cl.one_sample.classes{2},'2'));
            assert(cl.one_sample.classes_count == 2);
            assert(tc.check(cl.one_sample.samples == A(1,:)));
            assert(tc.check(cl.one_sample.labels_idx == c(1)));
            assert(cl.one_sample.samples_count == 1);
            assert(cl.one_sample.features_count == 2);
            assert(cl.one_sample.compatible(s));  
            assert(strcmp(cl.kernel_type,'poly'));
            assert(cl.kernel_param == 3);
            assert(strcmp(func2str(cl.svm_info.KernelFunction),'poly_kernel'));
            assert(cl.svm_info.KernelFunctionArgs{1} == 3);
            assert(tc.check(cl.svm_info.GroupNames == c));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With linear kernel and more than two classes possible.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);3*ones(100,1)];            
            s = dataset({'1' '2' '3' '4'},A,c);
            
            cl = classifiers.svm(s,'linear',0,log);
            
            assert(length(cl.one_sample.classes) == 4);
            assert(strcmp(cl.one_sample.classes{1},'1'));
            assert(strcmp(cl.one_sample.classes{2},'2'));
            assert(strcmp(cl.one_sample.classes{3},'3'));
            assert(strcmp(cl.one_sample.classes{4},'4'));
            assert(cl.one_sample.classes_count == 4);
            assert(tc.check(cl.one_sample.samples == A(1,:)));
            assert(tc.check(cl.one_sample.labels_idx == c(1)));
            assert(cl.one_sample.samples_count == 1);
            assert(cl.one_sample.features_count == 2);
            assert(cl.one_sample.compatible(s));  
            assert(strcmp(cl.kernel_type,'linear'));
            assert(cl.kernel_param == 0);
            assert(strcmp(func2str(cl.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.svm_info.KernelFunctionArgs));
            assert(tc.check(cl.svm_info.GroupNames == c));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);2*ones(100,1)];            
            s = dataset({'1' '2'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.svm(s_tr,'linear',0,log);            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts,log);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(tc.check(labels_confidence == ones(40,1)));
            assert(tc.check(labels_idx_hat2 == [repmat([1 2],20,1);repmat([2 1],20,1)]));
            assert(tc.check(labels_confidence2 == [ones(40,1),zeros(40,1)]));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n',...
                                                          'Computing dataset classes.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With mostly clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A_tr = [mvnrnd([3 1],[0.01 0; 0 0.01],80);
                    mvnrnd([1 3],[0.01 0; 0 0.01],80)];
            A_ts = [mvnrnd([3 1],[0.01 0; 0 0.01],19);
                    1 3;
                    mvnrnd([1 3],[0.01 0; 0 0.01],19)
                    3 1];
            c_tr = [1*ones(80,1);2*ones(80,1)];
            c_ts = [1*ones(20,1);2*ones(20,1)];            
            s_tr = dataset({'1' '2'},A_tr,c_tr);
            s_ts = dataset({'1' '2'},A_ts,c_ts);
            
            cl = classifiers.svm(s_tr,'linear',0,log);            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts,log);
            
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

            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n',...
                                                          'Computing dataset classes.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data and using a linear kernel.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1)];            
            s = dataset({'1' '2'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.svm(s_tr,'linear',0,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data and using an rbf kernel with sigma=0.5.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1)];            
            s = dataset({'1' '2'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.svm(s_tr,'rbf',0.5,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data and using an rbf kernel with sigma=0.1.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1)];            
            s = dataset({'1' '2'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.svm(s_tr,'rbf',0.1,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With clearly separated data and more than two classes possible.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);3*ones(100,1)];            
            s = dataset({'1' '2' '3' '4'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.svm(s_tr,'linear',0,log);
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts,log);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(tc.check(labels_confidence == ones(40,1)));
            assert(tc.check(labels_idx_hat2 == [repmat([1 2 3 4],20,1);repmat([3 2 1 4],20,1)]));
            assert(tc.check(labels_confidence2 == [ones(40,1),zeros(40,3)]));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n',...
                                                          'Computing dataset classes.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2' '3' '4'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end