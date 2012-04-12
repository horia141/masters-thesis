classdef one_vs_one < classifier
    properties (GetAccess=public,SetAccess=immutable)
        classifier_pairs;
        classifier_pairs_count;
        saved_classes;
    end
    
    methods (Access=public)
        function [obj] = one_vs_one(train_dataset,classifier_ctor_fn,params,logger)
            assert(tc.scalar(train_dataset));
            assert(tc.dataset(train_dataset));
            assert(tc.scalar(classifier_ctor_fn));
            assert(tc.function_h(classifier_ctor_fn));
            assert(tc.vector(params));
            assert(tc.cell(params));
            assert(tc.checkf(@tc.value,params) || ...
                   (tc.checkf(@tc.vector,params) && ...
                    (length(params) == 0.5 * train_dataset.classes_count * (train_dataset.classes_count - 1)) && ...
                    tc.checkf(@tc.cell,params) && ...
                    tc.checkf(@(c)tc.checkf(@tc.scalar,c),params) && ...
                    tc.checkf(@(c)tc.checkf(@tc.value,c),params)));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
                  
            if tc.check(cellfun(@tc.value,params))
                one_param_set = true;
            else
                one_param_set = false;
            end
            
            logger.beg_node('Training each classifier');
                
            current_classifier = 1;
            saved_classes_t = [];
                
            for ii = 1:train_dataset.classes_count
                for jj = (ii + 1):train_dataset.classes_count
                    if one_param_set
                        local_params = params;
                    else
                        local_params = params{current_classifier};
                    end
                    
                    try
                        local_dataset = train_dataset.subsamples(train_dataset.labels_idx == ii | train_dataset.labels_idx == jj);
                        classifier_pairs_t(current_classifier) = classifier_ctor_fn(local_dataset,local_params{:},logger.new_classifier('Classifier for %s vs %s',local_dataset.classes{ii},local_dataset.classes{jj}));
                        saved_classes_t = [saved_classes_t; ii jj];
                        current_classifier = current_classifier + 1;
                    catch exp
                        if strcmp(exp.identifier,'master:NoConvergence')
                            logger.end_node();
                            throw(MException('master:NoConvergence',exp.message));
                        else
                            rethrow(exp);
                        end
                    end
                end
            end
            
            logger.end_node();
            
            obj = obj@classifier(train_dataset.subsamples(1),logger);
            obj.classifier_pairs = classifier_pairs_t;
            obj.classifier_pairs_count = 0.5 * train_dataset.classes_count * (train_dataset.classes_count - 1);
            obj.saved_classes = saved_classes_t;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = do_classify(obj,dataset_d,logger)
            if obj.classifier_pairs_count == 1
                logger.beg_node('Classifying with each classifier');
                [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = obj.classifier_pairs(1).classify(dataset_d,...
                                                                                          logger.new_classifier('Classifier for %s vs %s',dataset_d.classes{1},dataset_d.classes{2}));
                logger.end_node();
            else
                partial_labels_idx = zeros(dataset_d.samples_count,obj.classifier_pairs_count);
                
                logger.beg_node('Classifying with each classifier');
                
                for ii = 1:obj.classifier_pairs_count
                    partial_labels_idx(:,ii) = obj.classifier_pairs(ii).classify(dataset_d,...
									  logger.new_classifier('Classifier for %s vs %s',dataset_d.classes{obj.saved_classes(ii,1)},dataset_d.classes{obj.saved_classes(ii,2)}));
                end

                logger.end_node();
                
                logger.message('Determining classes with most votes for each sample.');
                
                votes = hist(partial_labels_idx',dataset_d.classes_count);
                [~,labels_idx_hat] = max(votes,[],1);
                
                labels_idx_hat = labels_idx_hat';
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
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "classifiers.one_vs_one".\n');
            
            fprintf('  Testing proper construction.\n');
            
            fprintf('    With SVM and common parameters (linear kernel).\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1','2','3'},A,c);
            
            cl = classifiers.one_vs_one(s,@classifiers.svm,{'linear' 0},log);
            
            assert(length(cl.one_sample.classes) == 3);
            assert(strcmp(cl.one_sample.classes{1},'1'));
            assert(strcmp(cl.one_sample.classes{2},'2'));
            assert(strcmp(cl.one_sample.classes{3},'3'));
            assert(cl.one_sample.classes_count == 3);
            assert(tc.check(cl.one_sample.samples == A(1,:)));
            assert(tc.check(cl.one_sample.labels_idx == c(1)));
            assert(cl.one_sample.samples_count == 1);
            assert(cl.one_sample.features_count == 2);
            assert(cl.one_sample.compatible(s));
            assert(strcmp(func2str(cl.classifier_pairs(1).svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_pairs(1).svm_info.KernelFunctionArgs));
            assert(tc.check(cl.classifier_pairs(1).svm_info.GroupNames == c(c == 1 | c == 2)));
            assert(strcmp(cl.classifier_pairs(1).kernel_type,'linear'));
            assert(cl.classifier_pairs(1).kernel_param == 0);
            assert(strcmp(func2str(cl.classifier_pairs(2).svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_pairs(2).svm_info.KernelFunctionArgs));
            assert(tc.check(cl.classifier_pairs(2).svm_info.GroupNames == c(c == 1 | c == 3)));
            assert(strcmp(cl.classifier_pairs(2).kernel_type,'linear'));
            assert(cl.classifier_pairs(2).kernel_param == 0);
            assert(strcmp(func2str(cl.classifier_pairs(3).svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_pairs(3).svm_info.KernelFunctionArgs));
            assert(tc.check(cl.classifier_pairs(3).svm_info.GroupNames == c(c == 2 | c == 3)));
            assert(strcmp(cl.classifier_pairs(3).kernel_type,'linear'));
            assert(cl.classifier_pairs(3).kernel_param == 0);
            assert(cl.classifier_pairs_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs 2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1 vs 3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs 3:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVM and common parameters (rbf kernel with sigma=0.7).\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1','2','3'},A,c);
            
            cl = classifiers.one_vs_one(s,@classifiers.svm,{'rbf' 0.7},log);
            
            assert(length(cl.one_sample.classes) == 3);
            assert(strcmp(cl.one_sample.classes{1},'1'));
            assert(strcmp(cl.one_sample.classes{2},'2'));
            assert(strcmp(cl.one_sample.classes{3},'3'));
            assert(cl.one_sample.classes_count == 3);
            assert(tc.check(cl.one_sample.samples == A(1,:)));
            assert(tc.check(cl.one_sample.labels_idx == c(1)));
            assert(cl.one_sample.samples_count == 1);
            assert(cl.one_sample.features_count == 2);
            assert(cl.one_sample.compatible(s));
            assert(strcmp(func2str(cl.classifier_pairs(1).svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_pairs(1).svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.check(cl.classifier_pairs(1).svm_info.GroupNames == c(c == 1 | c == 2)));
            assert(strcmp(cl.classifier_pairs(1).kernel_type,'rbf'));
            assert(cl.classifier_pairs(1).kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifier_pairs(2).svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_pairs(2).svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.check(cl.classifier_pairs(2).svm_info.GroupNames == c(c == 1 | c == 3)));
            assert(strcmp(cl.classifier_pairs(2).kernel_type,'rbf'));
            assert(cl.classifier_pairs(2).kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifier_pairs(3).svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_pairs(3).svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.check(cl.classifier_pairs(3).svm_info.GroupNames == c(c == 2 | c == 3)));
            assert(strcmp(cl.classifier_pairs(3).kernel_type,'rbf'));
            assert(cl.classifier_pairs(3).kernel_param == 0.7);
            assert(cl.classifier_pairs_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs 2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1 vs 3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs 3:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVM and custom parameters.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1','2','3'},A,c);
            
            cl = classifiers.one_vs_one(s,@classifiers.svm,{{'linear' 0} {'rbf' 0.7} {'poly' 3}},log);
            
            assert(length(cl.one_sample.classes) == 3);
            assert(strcmp(cl.one_sample.classes{1},'1'));
            assert(strcmp(cl.one_sample.classes{2},'2'));
            assert(strcmp(cl.one_sample.classes{3},'3'));
            assert(cl.one_sample.classes_count == 3);
            assert(tc.check(cl.one_sample.samples == A(1,:)));
            assert(tc.check(cl.one_sample.labels_idx == c(1)));
            assert(cl.one_sample.samples_count == 1);
            assert(cl.one_sample.features_count == 2);
            assert(cl.one_sample.compatible(s));
            assert(strcmp(func2str(cl.classifier_pairs(1).svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_pairs(1).svm_info.KernelFunctionArgs));
            assert(tc.check(cl.classifier_pairs(1).svm_info.GroupNames == c(c == 1 | c == 2)));
            assert(strcmp(cl.classifier_pairs(1).kernel_type,'linear'));
            assert(cl.classifier_pairs(1).kernel_param == 0);
            assert(strcmp(func2str(cl.classifier_pairs(2).svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_pairs(2).svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.check(cl.classifier_pairs(2).svm_info.GroupNames == c(c == 1 | c == 3)));
            assert(strcmp(cl.classifier_pairs(2).kernel_type,'rbf'));
            assert(cl.classifier_pairs(2).kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifier_pairs(3).svm_info.KernelFunction),'poly_kernel'));
            assert(cl.classifier_pairs(3).svm_info.KernelFunctionArgs{1} == 3);
            assert(tc.check(cl.classifier_pairs(3).svm_info.GroupNames == c(c == 2 | c == 3)));
            assert(strcmp(cl.classifier_pairs(3).kernel_type,'poly'));
            assert(cl.classifier_pairs(3).kernel_param == 3);
            assert(cl.classifier_pairs_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs 2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1 vs 3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs 3:\n',...
                                                          '    Computing separation surfaces.\n'))));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With SVMs with linear kernels on clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([3 3],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.one_vs_one(s_tr,@classifiers.svm,{'linear' 0},log);
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts,log);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(tc.check(labels_confidence == ones(60,1)));
            assert(tc.check(labels_idx_hat2 == [repmat([1 2 3],20,1);repmat([2 1 3],20,1);repmat([3 2 1],20,1)]));
            assert(tc.check(labels_confidence2 == [ones(60,1),zeros(60,2)]));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs 2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1 vs 3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs 3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          'Classifying with each classifier:\n',...
                                                          '  Classifier for 1 vs 2:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 1 vs 3:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 2 vs 3:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          'Determining classes with most votes for each sample.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2' '3'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVMs with linear kernels on mostly clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
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
            s_tr = dataset({'1' '2' '3'},A_tr,c_tr);
            s_ts = dataset({'1' '2' '3'},A_ts,c_ts);
            
            cl = classifiers.one_vs_one(s_tr,@classifiers.svm,{'linear' 0},log);
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts,log);
            
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
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs 2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1 vs 3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs 3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          'Classifying with each classifier:\n',...
                                                          '  Classifier for 1 vs 2:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 1 vs 3:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 2 vs 3:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          'Determining classes with most votes for each sample.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2' '3'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVMs with linear kernels on not so clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([3 3],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.one_vs_one(s_tr,@classifiers.svm,{'linear' 0},log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs 2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1 vs 3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs 3:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2' '3'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVMs with rbf kernel and sigma=0.5 on not so clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([3 3],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.one_vs_one(s_tr,@classifiers.svm,{'rbf' 0.5},log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs 2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1 vs 3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs 3:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2' '3'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVMs with rbf kernel and sigma=0.1 on not so clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([3 3],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.one_vs_one(s_tr,@classifiers.svm,{'rbf' 0.1},log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs 2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1 vs 3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs 3:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2' '3'},ptmp,ones(length(ptmp),1)),log);
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
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
