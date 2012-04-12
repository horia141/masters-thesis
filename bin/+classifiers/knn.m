classdef knn < classifier
    properties (GetAccess=public,SetAccess=immutable)
        samples;
        k;
    end
    
    methods (Access=public)
        function [obj] = knn(train_dataset,k,logger)
            assert(tc.scalar(train_dataset));
            assert(tc.dataset(train_dataset));
            assert(train_dataset.samples_count >= 1);
            assert(tc.scalar(k));
            assert(tc.natural(k));
            assert(k >= 1);
            assert(k <= train_dataset.samples_count);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            logger.message('Storing dataset.');

            obj = obj@classifier(train_dataset.subsamples(1),logger);
            obj.samples = train_dataset;
            obj.k = k;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = do_classify(obj,dataset_d,logger)
            logger.message('Computing distances to each stored sample.');
            
            labels_idx_hat = knnclassify(dataset_d.samples,obj.samples.samples,obj.samples.labels_idx);
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
            fprintf('Testing "classifiers.knn".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1' '2' '3'},A,c);
            
            cl = classifiers.knn(s,1,log);
            
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
            assert(cl.samples == s);
            assert(cl.k == 1);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Storing dataset.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([3 3],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1' '2' '3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.knn(s_tr,3,log);            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts,log);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(tc.check(labels_confidence == ones(60,1)));
            assert(tc.check(labels_idx_hat2 == [repmat([1 2 3],20,1);repmat([2 1 3],20,1);repmat([3 2 1],20,1)]));
            assert(tc.check(labels_confidence2 == [ones(60,1),zeros(60,2)]));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Storing dataset.\n',...
                                                          'Computing distances to each stored sample.\n'))));
            
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
            
            fprintf('    With mostly clearly separated data.\n');
            
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
            
            cl = classifiers.knn(s_tr,3,log);            
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
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Storing dataset.\n',...
                                                          'Computing distances to each stored sample.\n'))));
            
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
            
            fprintf('    Without clearly separated data (k=3).\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.5 0; 0 0.5],100);
                 mvnrnd([3 3],[0.5 0; 0 0.5],100);
                 mvnrnd([1 3],[0.5 0; 0 0.5],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1' '2' '3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.knn(s_tr,3,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Storing dataset.\n'))));
            
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
            
            fprintf('    Without clearly separated data (k=7).\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.5 0; 0 0.5],100);
                 mvnrnd([3 3],[0.5 0; 0 0.5],100);
                 mvnrnd([1 3],[0.5 0; 0 0.5],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1' '2' '3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.knn(s_tr,7,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Storing dataset.\n'))));
            
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