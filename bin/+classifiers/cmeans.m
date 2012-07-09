classdef cmeans < classifier
    properties (GetAccess=public,SetAccess=immutable)
        centers;
    end

    methods (Access=public)
        function [obj] = cmeans(train_sample,class_info,logger)
            assert(check.dataset_record(train_sample));
            assert(check.scalar(class_info));
            assert(check.classifier_info(class_info));
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
            assert(logger.active);
            assert(class_info.compatible(train_sample));
            
            logger.message('Computing dataset class centers.');
            
            d = dataset.geometry(train_sample);
            centers_t = zeros(d,class_info.labels_count);
            
            for ii = 1:class_info.labels_count
                centers_t(:,ii) = mean(train_sample(:,class_info.labels_idx == ii),2);
            end
            
            input_geometry = d;
            
            obj = obj@classifier(input_geometry,class_info.labels,logger);
            obj.centers = centers_t;
        end
    end

    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            N = dataset.count(sample);

            distances = zeros(obj.saved_labels_count,N);
            
            logger.beg_node('Computing distances to each class center');
            
            for ii = 1:obj.saved_labels_count
                logger.message('Class %s.',obj.saved_labels{ii});
                
                distances(ii,:) = sum((sample - repmat(obj.centers(:,ii),1,N)) .^ 2,1);
            end
            
            logger.end_node();
            
            logger.message('Selecting class for each sample.');
            
            reciproc_distances = 1 ./ distances;
            [~,min_index] = max(reciproc_distances,[],1);

            labels_idx_hat = min_index;
            labels_confidence = bsxfun(@rdivide,reciproc_distances,sum(reciproc_distances,1));
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "classifiers.cmeans".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();

            cl = classifiers.cmeans(s,ci,logg);
            
            assert(check.checkv(arrayfun(@(ii)check.same(cl.centers(:,ii),mean(s(:,ci.labels_idx == ii),2)),1:3)));
            assert(check.same(cl.centers,[3 1;3 3;1 3]',0.1));
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_3();
            
            cl = classifiers.cmeans(s_tr,ci_tr,logg);            
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[3 60]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(check.same(conf_matrix,[20 0 0; 0 20 0; 0 0 20]));
            assert(check.empty(misclassified));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With mostly clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_mostly_clear_data_3();
            
            cl = classifiers.cmeans(s_tr,ci_tr,logg);            
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[3 60]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(check.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(check.same(misclassified,[19 20 39 40 59 60]));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Without clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});

            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_unclear_data_3();
            
            cl = classifiers.cmeans(s_tr,ci_tr,logg);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
        end
    end
end
