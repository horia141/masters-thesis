classdef logistic_regression < classifier
    properties (GetAccess=public,SetAccess=immutable)
        sep_plane_coeffs;
    end
    
    methods (Access=public)
        function [obj] = logistic_regression(train_sample,class_info,logger)
            assert(tc.dataset_record(train_sample));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(class_info.compatible(train_sample));
            
            logger.message('Computing separation surfaces.');
            
            input_geometry = dataset.geometry(train_sample);
            
            obj = obj@classifier(input_geometry,class_info.labels,logger);
            obj.sep_plane_coeffs = mnrfit(train_sample',class_info.labels_idx);
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            logger.message('Computing dataset classes.');
            
            classes_probs = mnrval(obj.sep_plane_coeffs,sample');
            [max_probs,max_probs_idx] = max(classes_probs,[],2);
            
            labels_idx_hat = max_probs_idx';
            labels_confidence = bsxfun(@rdivide,classes_probs,max_probs)';
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "classifiers.logistic_regression".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.logistic_regression(s,ci,log);
            
            assert(tc.same(cl.sep_plane_coeffs,mnrfit(s',ci.labels_idx)));
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.logistic_regression(s_tr,ci_tr,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.same(labels_confidence,[ones(20,1) zeros(20,1) zeros(20,1);zeros(20,1) ones(20,1) zeros(20,1);zeros(20,1) zeros(20,1) ones(20,1)]','Epsilon',1e-4));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
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
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.logistic_regression(s_tr,ci_tr,log);
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
            assert(tc.same(labels_confidence,[ones(18,1) zeros(18,1) zeros(18,1);0 1 0; 0 0 1;zeros(18,1) ones(18,1) zeros(18,1);1 0 0; 0 0 1;zeros(18,1) zeros(18,1) ones(18,1); 1 0 0; 0 1 0]','Epsilon',1e-5));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing separation surfaces.\n',...
                                                          'Computing dataset classes.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});

            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.logistic_regression(s_tr,ci_tr,log);
            
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
