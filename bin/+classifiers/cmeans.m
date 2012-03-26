classdef cmeans < classifier
    properties (GetAccess=public,SetAccess=immutable)
        centers;
    end

    methods (Access=public)
        function [obj] = cmeans(train_dataset)
            assert(tc.scalar(train_dataset) && tc.dataset(train_dataset));
            assert(train_dataset.samples_count >= 1);
            
            centers_t = zeros(train_dataset.classes_count,train_dataset.features_count);
            
            for i = 1:train_dataset.classes_count
                centers_t(i,:) = mean(train_dataset.samples(train_dataset.labels_idx == i,:));
            end
            
            obj = obj@classifier(train_dataset.subsamples(1));
            obj.centers = centers_t;
        end
    end

    methods (Access=protected)
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = do_classify(obj,dataset_d)
            distances = zeros(dataset_d.samples_count,obj.one_sample.classes_count);
            
            for i = 1:obj.one_sample.classes_count
                distances(:,i) = sum((dataset_d.samples - repmat(obj.centers(i,:),dataset_d.samples_count,1)) .^ 2,2);
            end
            
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
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1' '2' '3'},A,c);
            
            cl = classifiers.cmeans(s);
            
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
            assert(tc.check(arrayfun(@(i)utils.approx(cl.centers(i,:),mean(s.samples(s.labels_idx == i,:))),1:3)));
            assert(utils.approx(cl.centers,[3 1;3 3;1 3],0.1));
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([3 3],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1' '2' '3'},A,c);            
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);            
            t_centers = cell2mat(arrayfun(@(i)mean(s_tr.samples(s_tr.labels_idx == i,:))',1:3,'UniformOutput',false))';
            AA = cell2mat(arrayfun(@(i)sum((s_ts.samples - repmat(t_centers(i,:),60,1)) .^ 2,2),1:3,'UniformOutput',false));
            AA = bsxfun(@rdivide,AA',sum(AA,2)')';
            [AA_s,I_s] = sort(AA,2,'ascend');
            
            cl = classifiers.cmeans(s_tr);            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(utils.approx(labels_confidence,1 - AA_s(:,1)));
            assert(tc.check(labels_idx_hat2 == I_s));
            assert(utils.approx(labels_confidence2,1 - AA_s));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With mostly clearly separated data.\n');
            
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
            t_centers = cell2mat(arrayfun(@(i)mean(s_tr.samples(s_tr.labels_idx == i,:))',1:3,'UniformOutput',false))';
            AA = cell2mat(arrayfun(@(i)sum((s_ts.samples - repmat(t_centers(i,:),60,1)) .^ 2,2),1:3,'UniformOutput',false));
            AA = bsxfun(@rdivide,AA',sum(AA,2)')';
            [AA_s,I_s] = sort(AA,2,'ascend');
            
            cl = classifiers.cmeans(s_tr);            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts);
            
            assert(tc.check(labels_idx_hat(1:18) == s_ts.labels_idx(1:18)));
            assert(tc.check(labels_idx_hat(21:38) == s_ts.labels_idx(21:38)));
            assert(tc.check(labels_idx_hat(41:58) == s_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(utils.approx(labels_confidence,1 - AA_s(:,1)));
            assert(tc.check(labels_idx_hat2 == I_s));
            assert(utils.approx(labels_confidence2,1 - AA_s));
            assert(score == 90);
            assert(tc.check(conf_matrix == [18 1 1; 1 18 1; 1 1 18]));
            assert(tc.check(misclassified == [19 20 39 40 59 60]'));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data.\n');
            
            A = [mvnrnd([3 1],[0.5 0; 0 0.5],100);
                 mvnrnd([3 3],[0.5 0; 0 0.5],100);
                 mvnrnd([1 3],[0.5 0; 0 0.5],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];            
            s = dataset({'1' '2' '3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = classifiers.cmeans(s_tr);
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(dataset({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
