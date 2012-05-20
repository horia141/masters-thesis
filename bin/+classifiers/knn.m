classdef knn < classifier
    properties (GetAccess=public,SetAccess=immutable)
        sample;
        labels_idx;
        k;
    end
    
    methods (Access=public)
        function [obj] = knn(train_sample,class_info,k,logger)
            assert(tc.dataset_record(train_sample));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
            assert(tc.scalar(k));
            assert(tc.natural(k));
            assert(k >= 1);
            assert(k <= dataset.count(train_sample));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(class_info.compatible(train_sample));
            
            logger.message('Storing dataset.');
            
            input_geometry = dataset.geometry(train_sample);

            obj = obj@classifier(input_geometry,class_info.labels,logger);
            obj.sample = train_sample;
            obj.labels_idx = class_info.labels_idx;
            obj.k = k;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            logger.message('Computing distances to each stored sample.');
            
            N = dataset.count(sample);
            
            labels_idx_hat = knnclassify(sample',obj.sample',obj.labels_idx)';
            labels_confidence = zeros(obj.saved_labels_count,N);
            labels_confidence(sub2ind(size(labels_confidence),labels_idx_hat,1:N)) = 1;
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "classifiers.knn".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});

            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.knn(s,ci,1,log);
            
            assert(tc.same(cl.sample,s));
            assert(tc.same(cl.labels_idx,ci.labels_idx));
            assert(cl.k == 1);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Storing dataset.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.knn(s_tr,ci_tr,3,log);            
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.same(labels_confidence,[ones(20,1) zeros(20,1) zeros(20,1);zeros(20,1) ones(20,1) zeros(20,1);zeros(20,1) zeros(20,1) ones(20,1)]'));
            assert(score == 100);
            assert(tc.same(conf_matrix,[20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Storing dataset.\n',...
                                                          'Computing distances to each stored sample.\n'))));
            
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
            
            cl = classifiers.knn(s_tr,ci_tr,3,log);
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
            assert(tc.same(labels_confidence,[ones(18,1) zeros(18,1) zeros(18,1);0 1 0; 0 0 1;zeros(18,1) ones(18,1) zeros(18,1);1 0 0; 0 0 1;zeros(18,1) zeros(18,1) ones(18,1); 1 0 0; 0 1 0]'));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Storing dataset.\n',...
                                                          'Computing distances to each stored sample.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data (k=3).\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});

            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.knn(s_tr,ci_tr,3,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Storing dataset.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data (k=7).\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});

            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.knn(s_tr,ci_tr,7,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Storing dataset.\n'))));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
