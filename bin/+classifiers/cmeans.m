classdef cmeans < classifier
    properties (GetAccess=public,SetAccess=immutable)
        centers;
    end

    methods (Access=public)
        function [obj] = cmeans(train_dataset,logger)
            assert(tc.scalar(train_dataset));
            assert(tc.dataset(train_dataset));
            assert(train_dataset.samples_count >= 1);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            logger.message('Computing dataset class centers.');
            
            centers_t = zeros(train_dataset.classes_count,train_dataset.features_count);
            
            for ii = 1:train_dataset.classes_count
                centers_t(ii,:) = mean(train_dataset.samples(train_dataset.labels_idx == ii,:));
            end
            
            obj = obj@classifier(train_dataset.subsamples(1),logger);
            obj.centers = centers_t;
        end
    end

    methods (Access=protected)
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = do_classify(obj,dataset_d,logger)
            distances = zeros(dataset_d.samples_count,obj.one_sample.classes_count);
            
            logger.beg_node('Computing distances to each class center');
            
            for ii = 1:obj.one_sample.classes_count
                logger.message('Class %s.',dataset_d.classes{ii});
                
                distances(:,ii) = sum((dataset_d.samples - repmat(obj.centers(ii,:),dataset_d.samples_count,1)) .^ 2,2);
            end
            
            logger.end_node();
            
            logger.message('Selecting class for each sample.');
            
            normed_distances = bsxfun(@rdivide,distances,sum(distances,2));
            [sorted_distances,sorted_indices] = sort(normed_distances,2,'ascend');
            
            labels_idx_hat = sorted_indices(:,1);
            labels_confidence = 1 - sorted_distances(:,1);
            labels_idx_hat2 = sorted_indices;
            labels_confidence2 = 1 - sorted_distances;
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "classifiers.cmeans".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1' '2' '3'},A,c);
            
            cl = classifiers.cmeans(s,log);
            
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
            assert(tc.check(arrayfun(@(ii)tc.same(cl.centers(ii,:),mean(s.samples(s.labels_idx == ii,:))),1:3)));
            assert(tc.same(cl.centers,[3 1;3 3;1 3],'Epsilon',0.1));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset class centers.\n'))));
            
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
            t_centers = cell2mat(arrayfun(@(ii)mean(s_tr.samples(s_tr.labels_idx == ii,:))',1:3,'UniformOutput',false))';
            AA = cell2mat(arrayfun(@(ii)sum((s_ts.samples - repmat(t_centers(ii,:),60,1)) .^ 2,2),1:3,'UniformOutput',false));
            AA = bsxfun(@rdivide,AA',sum(AA,2)')';
            [AA_s,I_s] = sort(AA,2,'ascend');
            
            cl = classifiers.cmeans(s_tr,log);            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts,log);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(tc.same(labels_confidence,1 - AA_s(:,1)));
            assert(tc.check(labels_idx_hat2 == I_s));
            assert(tc.same(labels_confidence2,1 - AA_s));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset class centers.\n',...
                                                          'Computing distances to each class center:\n',...
                                                          '  Class 1.\n',...
                                                          '  Class 2.\n',...
                                                          '  Class 3.\n',...
                                                          'Selecting class for each sample.\n'))));
            
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
            t_centers = cell2mat(arrayfun(@(ii)mean(s_tr.samples(s_tr.labels_idx == ii,:))',1:3,'UniformOutput',false))';
            AA = cell2mat(arrayfun(@(ii)sum((s_ts.samples - repmat(t_centers(ii,:),60,1)) .^ 2,2),1:3,'UniformOutput',false));
            AA = bsxfun(@rdivide,AA',sum(AA,2)')';
            [AA_s,I_s] = sort(AA,2,'ascend');
            
            cl = classifiers.cmeans(s_tr,log);            
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
            assert(tc.same(labels_confidence,1 - AA_s(:,1)));
            assert(tc.check(labels_idx_hat2 == I_s));
            assert(tc.same(labels_confidence2,1 - AA_s));
            assert(score == 90);
            assert(tc.check(conf_matrix == [18 1 1; 1 18 1; 1 1 18]));
            assert(tc.check(misclassified == [19 20 39 40 59 60]'));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset class centers.\n',...
                                                          'Computing distances to each class center:\n',...
                                                          '  Class 1.\n',...
                                                          '  Class 2.\n',...
                                                          '  Class 3.\n',...
                                                          'Selecting class for each sample.\n'))));
            
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
            
            fprintf('    Without clearly separated data.\n');
            
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
            
            cl = classifiers.cmeans(s_tr,log);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset class centers.\n'))));

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
