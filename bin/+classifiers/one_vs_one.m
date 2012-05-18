classdef one_vs_one < classifier
    properties (GetAccess=public,SetAccess=immutable)
        classifier_list;
        classifier_count;
        saved_class_pair;
    end
    
    methods (Access=public)
        function [obj] = one_vs_one(train_sample,class_info,classifier_ctor_fn,params,logger)
            assert(tc.dataset_record(train_sample));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
            assert(tc.scalar(classifier_ctor_fn));
            assert(tc.function_h(classifier_ctor_fn));
            assert(tc.vector(params));
            assert(tc.cell(params));
            assert(tc.checkf(@tc.value,params) || ...
                   (tc.checkf(@tc.vector,params) && ...
                    (length(params) == 0.5 * class_info.labels_count * (class_info.labels_count - 1)) && ...
                    tc.checkf(@tc.cell,params) && ...
                    tc.checkf(@(c)tc.checkf(@tc.scalar,c),params) && ...
                    tc.checkf(@(c)tc.checkf(@tc.value,c),params)));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(class_info.compatible(train_sample));
                  
            classifier_count_t = 0.5 * class_info.labels_count * (class_info.labels_count - 1);
            
            if tc.check(cellfun(@tc.value,params))
                local_params = repmat({params},classifier_count_t,1);
            else
                local_params = params;
            end
            
            logger.beg_node('Training each classifier');
                
            classifier_list_t = cell(classifier_count_t,1);
            saved_class_pair_t = zeros(classifier_count_t,2);
            
            current_classifier = 1;
                
            try
                for ii = 1:class_info.labels_count
                    for jj = (ii + 1):class_info.labels_count
                        temp_labels_idx = class_info.labels_idx(class_info.labels_idx == ii | class_info.labels_idx == jj);
                        local_ci = classification_info({'One' 'Other'},(temp_labels_idx ~= ii) + 1);
                        local_sample = dataset.subsample(train_sample,class_info.labels_idx == ii | class_info.labels_idx == jj);
                        classifier_list_t{current_classifier} = classifier_ctor_fn(local_sample,local_ci,local_params{current_classifier}{:},logger.new_classifier('Classifier for %s-vs-%s',class_info.labels{ii},class_info.labels{jj}));
                        saved_class_pair_t(current_classifier,1) = ii;
                        saved_class_pair_t(current_classifier,2) = jj;
                        current_classifier = current_classifier + 1;
                    end
                end
            catch exp
                logger.end_node();

                if strcmp(exp.identifier,'master:NoConvergence')
                    throw(MException('master:NoConvergence',exp.message));
                else
                    rethrow(exp);
                end
            end

            logger.end_node();
            
            input_geometry = dataset.geometry(train_sample);
            
            obj = obj@classifier(input_geometry,class_info.labels,logger);
            obj.classifier_list = classifier_list_t;
            obj.classifier_count = classifier_count_t;
            obj.saved_class_pair = saved_class_pair_t;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            if obj.classifier_count == 1
                logger.beg_node('Classifying with each classifier');
                
                [labels_idx_hat,labels_confidence] = obj.classifier_lists{1}.classify(sample,-1,logger.new_classifier('Classifier for %s-vs-%s',obj.saved_labels{1},obj.saved_labels{2}));

                logger.end_node();
            else
                N = dataset.count(sample);
                pair_labels_idx = zeros(N,obj.classifier_count);
                partial_labels_idx = zeros(N,obj.classifier_count);
                
                logger.beg_node('Classifying with each classifier');
                
                for ii = 1:obj.classifier_count
                    pair_labels_idx(:,ii) = obj.classifier_list{ii}.classify(sample,-1,...
									  logger.new_classifier('Classifier for %s-vs-%s',obj.saved_labels{obj.saved_class_pair(ii,1)},obj.saved_labels{obj.saved_class_pair(ii,2)}));
                end
                
                for ii = 1:obj.classifier_count
                    partial_labels_idx(:,ii) = obj.saved_class_pair(ii,pair_labels_idx(:,ii))';
                end

                logger.end_node();
                
                logger.message('Determining classes with most votes for each sample.');
                
                votes = hist(partial_labels_idx',obj.saved_labels_count);
                [max_votes,labels_idx_hat_t] = max(votes,[],1);

                labels_idx_hat = labels_idx_hat_t';
                labels_confidence = bsxfun(@rdivide,votes,max_votes)';
            end
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "classifiers.one_vs_one".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With SVM and common parameters (linear kernel).\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.one_vs_one(s,ci,@classifiers.svm,{'linear' 0 1},log);
            
            tran_12 = [1 2 0];
            tran_13 = [1 0 2];
            tran_23 = [0 1 2];
            
            assert(strcmp(func2str(cl.classifier_list{1}.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_list{1}.svm_info.KernelFunctionArgs));
            assert(tc.same(cl.classifier_list{1}.svm_info.GroupNames,tran_12(ci.labels_idx(ci.labels_idx == 1 | ci.labels_idx == 2))'));
            assert(strcmp(cl.classifier_list{1}.kernel_type,'linear'));
            assert(cl.classifier_list{1}.kernel_param == 0);
            assert(strcmp(func2str(cl.classifier_list{2}.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_list{2}.svm_info.KernelFunctionArgs));
            assert(tc.same(cl.classifier_list{2}.svm_info.GroupNames,tran_13(ci.labels_idx(ci.labels_idx == 1 | ci.labels_idx == 3))'));
            assert(strcmp(cl.classifier_list{2}.kernel_type,'linear'));
            assert(cl.classifier_list{2}.kernel_param == 0);
            assert(strcmp(func2str(cl.classifier_list{3}.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_list{3}.svm_info.KernelFunctionArgs));
            assert(tc.same(cl.classifier_list{3}.svm_info.GroupNames,tran_23(ci.labels_idx(ci.labels_idx == 2 | ci.labels_idx == 3))'));
            assert(strcmp(cl.classifier_list{3}.kernel_type,'linear'));
            assert(cl.classifier_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1';'2';'3'}));
            assert(cl.saved_labels_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1-vs-2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1-vs-3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2-vs-3:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVM and common parameters (rbf kernel with sigma=0.7).\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.one_vs_one(s,ci,@classifiers.svm,{'rbf' 0.7 1},log);
            
            tran_12 = [1 2 0];
            tran_13 = [1 0 2];
            tran_23 = [0 1 2];
            
            assert(strcmp(func2str(cl.classifier_list{1}.svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_list{1}.svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.same(cl.classifier_list{1}.svm_info.GroupNames,tran_12(ci.labels_idx(ci.labels_idx == 1 | ci.labels_idx == 2))'));
            assert(strcmp(cl.classifier_list{1}.kernel_type,'rbf'));
            assert(cl.classifier_list{1}.kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifier_list{2}.svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_list{2}.svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.same(cl.classifier_list{2}.svm_info.GroupNames,tran_13(ci.labels_idx(ci.labels_idx == 1 | ci.labels_idx == 3))'));
            assert(strcmp(cl.classifier_list{2}.kernel_type,'rbf'));
            assert(cl.classifier_list{2}.kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifier_list{3}.svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_list{3}.svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.same(cl.classifier_list{3}.svm_info.GroupNames,tran_23(ci.labels_idx(ci.labels_idx == 2 | ci.labels_idx == 3))'));
            assert(strcmp(cl.classifier_list{3}.kernel_type,'rbf'));
            assert(cl.classifier_list{3}.kernel_param == 0.7);
            assert(cl.classifier_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1';'2';'3'}));
            assert(cl.saved_labels_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1-vs-2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1-vs-3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2-vs-3:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVM and custom parameters.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.one_vs_one(s,ci,@classifiers.svm,{{'linear' 0 1} {'rbf' 0.7 1} {'poly' 3 1}},log);
            
            tran_12 = [1 2 0];
            tran_13 = [1 0 2];
            tran_23 = [0 1 2];
            
            assert(strcmp(func2str(cl.classifier_list{1}.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_list{1}.svm_info.KernelFunctionArgs));
            assert(tc.same(cl.classifier_list{1}.svm_info.GroupNames,tran_12(ci.labels_idx(ci.labels_idx == 1 | ci.labels_idx == 2))'));
            assert(strcmp(cl.classifier_list{1}.kernel_type,'linear'));
            assert(cl.classifier_list{1}.kernel_param == 0);
            assert(strcmp(func2str(cl.classifier_list{2}.svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_list{2}.svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.same(cl.classifier_list{2}.svm_info.GroupNames,tran_13(ci.labels_idx(ci.labels_idx == 1 | ci.labels_idx == 3))'));
            assert(strcmp(cl.classifier_list{2}.kernel_type,'rbf'));
            assert(cl.classifier_list{2}.kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifier_list{3}.svm_info.KernelFunction),'poly_kernel'));
            assert(cl.classifier_list{3}.svm_info.KernelFunctionArgs{1} == 3);
            assert(tc.same(cl.classifier_list{3}.svm_info.GroupNames,tran_23(ci.labels_idx(ci.labels_idx == 2 | ci.labels_idx == 3))'));
            assert(strcmp(cl.classifier_list{3}.kernel_type,'poly'));
            assert(cl.classifier_list{3}.kernel_param == 3);
            assert(cl.classifier_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1';'2';'3'}));
            assert(cl.saved_labels_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1-vs-2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1-vs-3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2-vs-3:\n',...
                                                          '    Computing separation surfaces.\n'))));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With SVMs with linear kernels on clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.one_vs_one(s_tr,ci_tr,@classifiers.svm,{'linear' 0 1},log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.same(labels_confidence(1:20,1),ones(20,1)));
            assert(tc.check(labels_confidence(1:20,2:3) < 1));
            assert(tc.same(labels_confidence(21:40,2),ones(20,1)));
            assert(tc.check(labels_confidence(21:40,[1 3]) < 1));
            assert(tc.same(labels_confidence(41:60,3),ones(20,1)));
            assert(tc.check(labels_confidence(41:60,1:2) < 1));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1-vs-2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1-vs-3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2-vs-3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          'Classifying with each classifier:\n',...
                                                          '  Classifier for 1-vs-2:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 1-vs-3:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 2-vs-3:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          'Determining classes with most votes for each sample.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVMs with linear kernels on mostly clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.one_vs_one(s_tr,ci_tr,@classifiers.svm,{'linear' 0 1},log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(tc.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(tc.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(tc.same(labels_confidence(1:18,1),ones(18,1)));
            assert(tc.same(labels_confidence(19,2),1));
            assert(tc.same(labels_confidence(20,3),1));
            assert(tc.check(labels_confidence(1:18,2:3) < 1));
            assert(tc.check(labels_confidence(19,[1 3]) < 1));
            assert(tc.check(labels_confidence(20,1:2) < 1));
            assert(tc.same(labels_confidence(21:38,2),ones(18,1)));
            assert(tc.same(labels_confidence(39,1),1));
            assert(tc.same(labels_confidence(40,3),1));
            assert(tc.check(labels_confidence(21:38,[1 3]) < 1));
            assert(tc.check(labels_confidence(39,[2 3]) < 1));
            assert(tc.check(labels_confidence(40,1:2) < 1));
            assert(tc.same(labels_confidence(41:58,3),ones(18,1)));
            assert(tc.same(labels_confidence(59,1),1));
            assert(tc.same(labels_confidence(60,2),1));
            assert(tc.check(labels_confidence(41:58,1:2) < 1));
            assert(tc.check(labels_confidence(59,2:3) < 1));
            assert(tc.check(labels_confidence(60,[1 3]) < 1));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]'));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1-vs-2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1-vs-3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2-vs-3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          'Classifying with each classifier:\n',...
                                                          '  Classifier for 1-vs-2:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 1-vs-3:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 2-vs-3:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          'Determining classes with most votes for each sample.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVMs with linear kernels on not so clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.one_vs_one(s_tr,ci_tr,@classifiers.svm,{'linear' 0 1},log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1-vs-2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1-vs-3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2-vs-3:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVMs with rbf kernel and sigma=0.5 on not so clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.one_vs_one(s_tr,ci_tr,@classifiers.svm,{'rbf' 0.5 1},log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1-vs-2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1-vs-3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2-vs-3:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVMs with rbf kernel and sigma=0.1 on not so clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.one_vs_one(s_tr,ci_tr,@classifiers.svm,{'rbf' 0.1 1},log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1-vs-2:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 1-vs-3:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2-vs-3:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
