classdef svm < classifiers.binary
    properties (GetAccess=public,SetAccess=immutable)
        svm_info;
        kernel_type;
        kernel_param;
        reg_param;
    end
    
    methods (Access=public)
        function [obj] = svm(train_sample,class_info,kernel_type,kernel_param,reg_param,logger)
            assert(tc.dataset_record(train_sample));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
            assert(class_info.labels_count == 2);
            assert(tc.scalar(kernel_type));
            assert(tc.string(kernel_type));
            assert(tc.one_of(kernel_type,'linear','rbf','poly'));
            assert((strcmp(kernel_type,'linear') && ...
                     tc.scalar(kernel_param) && tc.number(kernel_param) && (kernel_param == 0)) || ...
                   (strcmp(kernel_type,'rbf') && ...
                     tc.scalar(kernel_param) && tc.number(kernel_param) && (kernel_param > 0)) || ...
                   (strcmp(kernel_type,'poly') && ...
                     tc.scalar(kernel_param) && tc.natural(kernel_param) && (kernel_param > 0)));
            assert(tc.scalar(reg_param));
            assert(tc.number(reg_param));
            assert(reg_param > 0);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(class_info.compatible(train_sample));

            logger.message('Computing separation surfaces.');
            
            N = dataset.count(train_sample);

            try
                if strcmp(kernel_type,'linear')
                    svm_info_t = svmtrain(train_sample',class_info.labels_idx,'kernel_function','linear','boxconstraint',reg_param);
                    kernel_param_t = 0;
                elseif strcmp(kernel_type,'rbf')
                    svm_info_t = svmtrain(train_sample',class_info.labels_idx,'kernel_function','rbf','rbf_sigma',kernel_param,'kernelcachelimit',N,'boxconstraint',reg_param);
                    kernel_param_t = kernel_param;
                elseif strcmp(kernel_type,'poly')
                    svm_info_t = svmtrain(train_sample',class_info.labels_idx,'kernel_function','poly','polyorder',kernel_param,'kernelcachelimit',N,'boxconstraint',reg_param);
                    kernel_param_t = kernel_param;
                else
                    assert(false);
                end
            catch exp
                if strcmp(exp.identifier,'Bioinfo:svmtrain:NoConvergence')
                    throw(MException('master:NoConvergence',...
                             sprintf('Failed to converge with kernel "%s" with parameter "%f" and "C=%f"',kernel_type,kernel_param,reg_param)));
                else
                    rethrow(exp);
                end
            end
            
            input_geometry = dataset.geometry(train_sample);

            obj = obj@classifiers.binary(input_geometry,class_info.labels,logger);
            obj.svm_info = svm_info_t;
            obj.kernel_type = kernel_type;
            obj.kernel_param = kernel_param_t;
            obj.reg_param = reg_param;
        end
    end
       
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            logger.message('Computing dataset classes.');
            
            N = dataset.count(sample);

            labels_idx_hat = svmclassify(obj.svm_info,sample')';
            labels_confidence = zeros(obj.saved_labels_count,N);
            labels_confidence(sub2ind(size(labels_confidence),labels_idx_hat,1:N)) = 1;
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "classifiers.svm".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With linear kernel and param=0.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_2();
            
            cl = classifiers.svm(s,ci,'linear',0,1,log);
            
            assert(strcmp(cl.kernel_type,'linear'));
            assert(cl.kernel_param == 0);
            assert(strcmp(func2str(cl.svm_info.KernelFunction),'linear_kernel'));
            assert(tc.empty(cl.svm_info.KernelFunctionArgs));
            assert(tc.same(cl.svm_info.GroupNames,ci.labels_idx'));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With rbf kernel.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_2();
            
            cl = classifiers.svm(s,ci,'rbf',0.7,1,log);
            
            assert(strcmp(cl.kernel_type,'rbf'));
            assert(cl.kernel_param == 0.7);
            assert(strcmp(func2str(cl.svm_info.KernelFunction),'rbf_kernel'));
            assert(tc.same(cl.svm_info.KernelFunctionArgs,{0.7}));
            assert(tc.same(cl.svm_info.GroupNames,ci.labels_idx'));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With poly kernel.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_2();
            
            cl = classifiers.svm(s,ci,'poly',3,1,log);
            
            assert(strcmp(cl.kernel_type,'poly'));
            assert(cl.kernel_param == 3);
            assert(strcmp(func2str(cl.svm_info.KernelFunction),'poly_kernel'));
            assert(tc.same(cl.svm_info.KernelFunctionArgs,{3}));
            assert(tc.check(cl.svm_info.GroupNames,ci.labels_idx'));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm(s_tr,ci_tr,'linear',0,1,log);            
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.same(labels_confidence,[ones(20,1) zeros(20,1);zeros(20,1) ones(20,1)]'));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n',...
                                                          'Computing dataset classes.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With mostly clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});

            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_2();

            cl = classifiers.svm(s_tr,ci_tr,'linear',0,1,log);            
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat(1:19),ci_ts.labels_idx(1:19)));
            assert(tc.same(labels_idx_hat(21:39),ci_ts.labels_idx(21:39)));
            assert(labels_idx_hat(20) == 2);
            assert(labels_idx_hat(40) == 1);
            assert(tc.same(labels_confidence,[ones(19,1) zeros(19,1);0 1;zeros(19,1) ones(19,1);1 0]'));
            assert(score == 95);
            assert(tc.same(conf_matrix,[19 1; 1 19]));
            assert(tc.same(misclassified,[20 40]));

            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n',...
                                                          'Computing dataset classes.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data and using a linear kernel.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_2();
            
            cl = classifiers.svm(s_tr,ci_tr,'linear',0,1,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data and using an rbf kernel with sigma=0.5.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_2();
            
            cl = classifiers.svm(s_tr,ci_tr,'rbf',0.5,1,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data and using an rbf kernel with sigma=0.1.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_2();
            
            cl = classifiers.svm(s_tr,ci_tr,'rbf',0.1,1,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end