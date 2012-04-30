classdef cmeans < classifier
    properties (GetAccess=public,SetAccess=immutable)
        centers;
    end

    methods (Access=public)
        function [obj] = cmeans(train_sample,class_info,logger)
            assert(tc.dataset_record(train_sample));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(class_info.compatible(train_sample));
            
            logger.message('Computing dataset class centers.');
            
            d = dataset.geometry(train_sample);
            centers_t = zeros(class_info.labels_count,d);
            
            for ii = 1:class_info.labels_count
                centers_t(ii,:) = mean(train_sample(class_info.labels_idx == ii,:));
            end
            
            input_geometry = d;
            
            obj = obj@classifier(input_geometry,class_info.labels,logger);
            obj.centers = centers_t;
        end
    end

    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            N = dataset.count(sample);

            distances = zeros(N,obj.saved_labels_count);
            
            logger.beg_node('Computing distances to each class center');
            
            for ii = 1:obj.saved_labels_count
                logger.message('Class %s.',obj.saved_labels{ii});
                
                distances(:,ii) = sum((sample - repmat(obj.centers(ii,:),N,1)) .^ 2,2);
            end
            
            logger.end_node();
            
            logger.message('Selecting class for each sample.');
            
            [min_distance,min_index] = min(distances,[],2);

            labels_idx_hat = min_index(:,1);
            labels_confidence = 1 ./ bsxfun(@rdivide,distances,min_distance);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "classifiers.cmeans".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();

            cl = classifiers.cmeans(s,ci,log);
            
            assert(tc.check(arrayfun(@(ii)tc.same(cl.centers(ii,:),mean(s(ci.labels_idx == ii,:))),1:3)));
            assert(tc.same(cl.centers,[3 1;3 3;1 3],'Epsilon',0.1));
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1';'2';'3'}));
            assert(cl.saved_labels_count == 3);

            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset class centers.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            t_centers = cell2mat(arrayfun(@(ii)mean(s_tr(ci_tr.labels_idx == ii,:))',1:3,'UniformOutput',false))';
            dists = cell2mat(arrayfun(@(ii)sum((s_ts - repmat(t_centers(ii,:),60,1)) .^ 2,2),1:3,'UniformOutput',false));
            min_dists = min(dists,[],2);
            distsn = 1 ./ bsxfun(@rdivide,dists,min_dists);
            
            cl = classifiers.cmeans(s_tr,ci_tr,log);            
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.same(labels_confidence,distsn));
            assert(score == 100);
            assert(tc.same(conf_matrix,[20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset class centers.\n',...
                                                          'Computing distances to each class center:\n',...
                                                          '  Class 1.\n',...
                                                          '  Class 2.\n',...
                                                          '  Class 3.\n',...
                                                          'Selecting class for each sample.\n'))));
            
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
            
            t_centers = cell2mat(arrayfun(@(ii)mean(s_tr(ci_tr.labels_idx == ii,:))',1:3,'UniformOutput',false))';
            dists = cell2mat(arrayfun(@(ii)sum((s_ts - repmat(t_centers(ii,:),60,1)) .^ 2,2),1:3,'UniformOutput',false));
            min_dists = min(dists,[],2);
            distsn = 1 ./ bsxfun(@rdivide,dists,min_dists);
            
            cl = classifiers.cmeans(s_tr,ci_tr,log);            
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
            assert(tc.same(labels_confidence,distsn));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]'));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset class centers.\n',...
                                                          'Computing distances to each class center:\n',...
                                                          '  Class 1.\n',...
                                                          '  Class 2.\n',...
                                                          '  Class 3.\n',...
                                                          'Selecting class for each sample.\n'))));
            
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
            
            cl = classifiers.cmeans(s_tr,ci_tr,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset class centers.\n'))));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
