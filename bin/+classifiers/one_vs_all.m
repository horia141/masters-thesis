classdef one_vs_all < classifier
    properties (GetAccess=public,SetAccess=immutable)
        classifier_list;
        classifier_count;
    end
    
    methods (Access=public)
        function [obj] = one_vs_all(train_sample,class_info,classifier_ctor_fn,params,logger)
            assert(tc.dataset_record(train_sample));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
            assert(tc.scalar(classifier_ctor_fn));
            assert(tc.function_h(classifier_ctor_fn));
            assert(tc.vector(params));
            assert(tc.cell(params));
            assert(tc.checkf(@tc.value,params) || ...
                    (tc.checkf(@tc.vector,params) && ...
                      (length(params) == class_info.labels_count ) && ...
                      tc.checkf(@tc.cell,params) && ...
                      tc.checkf(@(c)tc.checkf(@tc.scalar,c),params) && ...
                      tc.checkf(@(c)tc.checkf(@tc.value,c),params)));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(class_info.compatible(train_sample));
            
            if tc.check(cellfun(@tc.value,params))
                local_params = repmat({params},class_info.labels_count,1);
            else
                local_params = params;
            end
            
            logger.beg_node('Training each classifier');
            
            classifier_list_t = cell(class_info.labels_count,1);
            
            try
                for ii = 1:class_info.labels_count
                    local_ci = classification_info({'one' 'rest'},(class_info.labels_idx ~= ii) + 1);
                    classifier_list_t{ii} = classifier_ctor_fn(train_sample,local_ci,local_params{ii}{:},logger.new_classifier('Classifier for %s vs rest',class_info.labels{ii}));
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
            obj.classifier_count = class_info.labels_count;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            logger.beg_node('Classifying with each classifier');

            N = dataset.count(sample);
            classes_probs = zeros(N,obj.saved_labels_count);
            
            for ii = 1:obj.classifier_count
                [~,local_confidence] = obj.classifier_list{ii}.classify(sample,-1,logger.new_classifier('Classifier for %s vs rest',obj.saved_labels{ii}));
                classes_probs(:,ii) = local_confidence(:,1);
            end
            
            logger.end_node();
            
            logger.message('Determining most probable class.');
            
            [max_probs,max_probs_idx] = max(classes_probs,[],2);
            
            labels_idx_hat = max_probs_idx;
            labels_confidence = bsxfun(@rdivide,classes_probs,max_probs);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "classifiers.one_vs_all".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With SVM and common parameters (linear kernel).\n');

            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.one_vs_all(s,ci,@classifiers.svm,{'linear' 0},log);
            
            assert(strcmp(func2str(cl.classifier_list{1}.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_list{1}.svm_info.KernelFunctionArgs));
            assert(tc.same(cl.classifier_list{1}.svm_info.GroupNames,(ci.labels_idx ~= 1) + 1));
            assert(strcmp(cl.classifier_list{1}.kernel_type,'linear'));
            assert(cl.classifier_list{1}.kernel_param == 0);
            assert(strcmp(func2str(cl.classifier_list{2}.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_list{2}.svm_info.KernelFunctionArgs));
            assert(tc.same(cl.classifier_list{2}.svm_info.GroupNames,(ci.labels_idx ~= 2) + 1));
            assert(strcmp(cl.classifier_list{2}.kernel_type,'linear'));
            assert(cl.classifier_list{2}.kernel_param == 0);
            assert(strcmp(func2str(cl.classifier_list{3}.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_list{3}.svm_info.KernelFunctionArgs));
            assert(tc.same(cl.classifier_list{3}.svm_info.GroupNames,(ci.labels_idx ~= 3) + 1));
            assert(strcmp(cl.classifier_list{3}.kernel_type,'linear'));
            assert(cl.classifier_count == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1';'2';'3'}));
            assert(cl.saved_labels_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 3 vs rest:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVM and common parameters (rbf kernel with sigma=0.7).\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.one_vs_all(s,ci,@classifiers.svm,{'rbf' 0.7},log);
            
            assert(strcmp(func2str(cl.classifier_list{1}.svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_list{1}.svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.same(cl.classifier_list{1}.svm_info.GroupNames,(ci.labels_idx ~= 1) + 1));
            assert(strcmp(cl.classifier_list{1}.kernel_type,'rbf'));
            assert(cl.classifier_list{1}.kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifier_list{2}.svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_list{2}.svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.same(cl.classifier_list{2}.svm_info.GroupNames,(ci.labels_idx ~= 2) + 1));
            assert(strcmp(cl.classifier_list{2}.kernel_type,'rbf'));
            assert(cl.classifier_list{2}.kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifier_list{3}.svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_list{3}.svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.same(cl.classifier_list{3}.svm_info.GroupNames,(ci.labels_idx ~= 3) + 1));
            assert(strcmp(cl.classifier_list{3}.kernel_type,'rbf'));
            assert(cl.classifier_list{3}.kernel_param == 0.7);
            assert(cl.classifier_count == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1';'2';'3'}));
            assert(cl.saved_labels_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 3 vs rest:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With SVM and custom parameters.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.one_vs_all(s,ci,@classifiers.svm,{{'linear' 0} {'rbf' 0.7} {'poly' 3}},log);
            
            assert(strcmp(func2str(cl.classifier_list{1}.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.classifier_list{1}.svm_info.KernelFunctionArgs));
            assert(tc.same(cl.classifier_list{1}.svm_info.GroupNames,(ci.labels_idx ~= 1) + 1));
            assert(strcmp(cl.classifier_list{1}.kernel_type,'linear'));
            assert(cl.classifier_list{1}.kernel_param == 0);
            assert(strcmp(func2str(cl.classifier_list{2}.svm_info.KernelFunction),'rbf_kernel'));
            assert(cl.classifier_list{2}.svm_info.KernelFunctionArgs{1} == 0.7);
            assert(tc.same(cl.classifier_list{2}.svm_info.GroupNames,(ci.labels_idx ~= 2) + 1));
            assert(strcmp(cl.classifier_list{2}.kernel_type,'rbf'));
            assert(cl.classifier_list{2}.kernel_param == 0.7);
            assert(strcmp(func2str(cl.classifier_list{3}.svm_info.KernelFunction),'poly_kernel'));
            assert(cl.classifier_list{3}.svm_info.KernelFunctionArgs{1} == 3);
            assert(tc.same(cl.classifier_list{3}.svm_info.GroupNames,(ci.labels_idx ~= 3) + 1));
            assert(strcmp(cl.classifier_list{3}.kernel_type,'poly'));
            assert(cl.classifier_list{3}.kernel_param == 3);
            assert(cl.classifier_count == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1';'2';'3'}));
            assert(cl.saved_labels_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 3 vs rest:\n',...
                                                          '    Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With SVMs with linear kernels on clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.one_vs_all(s_tr,ci_tr,@classifiers.svm,{'linear' 0},log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.same(labels_confidence,[ones(20,1) zeros(20,1) zeros(20,1);zeros(20,1) ones(20,1) zeros(20,1);zeros(20,1) zeros(20,1) ones(20,1)],'Epsilon',1e-4));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 3 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          'Classifying with each classifier:\n',...
                                                          '  Classifier for 1 vs rest:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 2 vs rest:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 3 vs rest:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          'Determining most probable class.\n'))));
            
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
            
            cl = classifiers.one_vs_all(s_tr,ci_tr,@classifiers.svm,{'linear' 0},log);
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
            assert(tc.same(labels_confidence,[ones(18,1) zeros(18,1) zeros(18,1);0 1 0; 0 0 1;zeros(18,1) ones(18,1) zeros(18,1);1 0 0; 0 0 1;zeros(18,1) zeros(18,1) ones(18,1); 1 0 0; 0 1 0],'Epsilon',1e-5));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]'));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 3 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          'Classifying with each classifier:\n',...
                                                          '  Classifier for 1 vs rest:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 2 vs rest:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          '  Classifier for 3 vs rest:\n',...
                                                          '    Computing dataset classes.\n',...
                                                          'Determining most probable class.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            clearvars -except display;
            
            fprintf('    With SVMs with linear kernels on not so clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.one_vs_all(s_tr,ci_tr,@classifiers.svm,{'linear' 0},log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 3 vs rest:\n',...
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
            
            cl = classifiers.one_vs_all(s_tr,ci_tr,@classifiers.svm,{'rbf' 0.5},log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 3 vs rest:\n',...
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
            
            cl = classifiers.one_vs_all(s_tr,ci_tr,@classifiers.svm,{'rbf' 0.1},log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Training each classifier:\n',...
                                                          '  Classifier for 1 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 2 vs rest:\n',...
                                                          '    Computing separation surfaces.\n',...
                                                          '  Classifier for 3 vs rest:\n',...
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
