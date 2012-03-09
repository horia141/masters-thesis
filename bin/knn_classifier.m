classdef knn_classifier < classifier
    properties (GetAccess=public,SetAccess=immutable)
        samples;
        k;
    end
    
    methods (Access=public)
        function [obj] = knn_classifier(train_samples,k)
            assert(tc.scalar(train_samples) && tc.samples_set(train_samples));
            assert(train_samples.samples_count > 0);
            assert(tc.scalar(k) && tc.natural(k) && (k > 0));
        
            obj = obj@classifier(train_samples);
            obj.samples = train_samples;
            obj.k = k;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat] = do_classify(obj,samples)
            labels_idx_hat = knnclassify(samples.samples,obj.samples.samples,obj.samples.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "knn_classifier".\n');
            
            fprintf('  Proper construction.\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            cl = knn_classifier(s,1);
            
            assert(cl.samples == s);
            assert(cl.k == 1);
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([3 3],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = knn_classifier(s_tr,3);
            
            [labels_idx_hat,score,conf_matrix,misclassified] = cl.classify(s_ts);
            
            assert(length(labels_idx_hat) == s_ts.samples_count);
            assert(all(labels_idx_hat == s_ts.labels_idx));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                scatter(s_tr.samples(s_tr.labels_idx == 1,1),s_tr.samples(s_tr.labels_idx == 1,2),'x','r');
                scatter(s_tr.samples(s_tr.labels_idx == 2,1),s_tr.samples(s_tr.labels_idx == 2,2),'x','g');
                scatter(s_tr.samples(s_tr.labels_idx == 3,1),s_tr.samples(s_tr.labels_idx == 3,2),'x','b');
                scatter(s_ts.samples(labels_idx_hat == 1,1),s_ts.samples(labels_idx_hat == 1,2),'o','r');
                scatter(s_ts.samples(labels_idx_hat == 2,1),s_ts.samples(labels_idx_hat == 2,2),'o','g');
                scatter(s_ts.samples(labels_idx_hat == 3,1),s_ts.samples(labels_idx_hat == 3,2),'o','b');
                ptmp = allcomb(0:0.1:4,0:0.1:4);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([0 4 0 4]);                
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
            
            s_tr = samples_set({'1' '2' '3'},A_tr,c_tr);
            s_ts = samples_set({'1' '2' '3'},A_ts,c_ts);
            cl = knn_classifier(s_tr,3);
            
            [labels_idx_hat,score,conf_matrix,misclassified] = cl.classify(s_ts);
            
            assert(length(labels_idx_hat) == s_ts.samples_count);
            assert(tc.check(labels_idx_hat(1:18) == s_ts.labels_idx(1:18)));
            assert(tc.check(labels_idx_hat(21:38) == s_ts.labels_idx(21:38)));
            assert(tc.check(labels_idx_hat(41:58) == s_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(score == 90);
            assert(tc.check(conf_matrix == [18 1 1; 1 18 1; 1 1 18]));
            assert(tc.check(misclassified == [19 20 39 40 59 60]'));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                scatter(s_tr.samples(s_tr.labels_idx == 1,1),s_tr.samples(s_tr.labels_idx == 1,2),'x','r');
                scatter(s_tr.samples(s_tr.labels_idx == 2,1),s_tr.samples(s_tr.labels_idx == 2,2),'x','g');
                scatter(s_tr.samples(s_tr.labels_idx == 3,1),s_tr.samples(s_tr.labels_idx == 3,2),'x','b');
                scatter(s_ts.samples(labels_idx_hat == 1,1),s_ts.samples(labels_idx_hat == 1,2),'o','r');
                scatter(s_ts.samples(labels_idx_hat == 2,1),s_ts.samples(labels_idx_hat == 2,2),'o','g');
                scatter(s_ts.samples(labels_idx_hat == 3,1),s_ts.samples(labels_idx_hat == 3,2),'o','b');
                ptmp = allcomb(0:0.1:4,0:0.1:4);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([0 4 0 4]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data (k=3).\n');
            
            A = [mvnrnd([3 1],[0.5 0; 0 0.5],100);
                 mvnrnd([3 3],[0.5 0; 0 0.5],100);
                 mvnrnd([1 3],[0.5 0; 0 0.5],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = knn_classifier(s_tr,3);
            
            [labels_idx_hat,~,~,~] = cl.classify(s_ts);
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                scatter(s_tr.samples(s_tr.labels_idx == 1,1),s_tr.samples(s_tr.labels_idx == 1,2),'x','r');
                scatter(s_tr.samples(s_tr.labels_idx == 2,1),s_tr.samples(s_tr.labels_idx == 2,2),'x','g');
                scatter(s_tr.samples(s_tr.labels_idx == 3,1),s_tr.samples(s_tr.labels_idx == 3,2),'x','b');
                scatter(s_ts.samples(labels_idx_hat == 1,1),s_ts.samples(labels_idx_hat == 1,2),'o','r');
                scatter(s_ts.samples(labels_idx_hat == 2,1),s_ts.samples(labels_idx_hat == 2,2),'o','g');
                scatter(s_ts.samples(labels_idx_hat == 3,1),s_ts.samples(labels_idx_hat == 3,2),'o','b');
                ptmp = allcomb(0:0.1:4,0:0.1:4);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([0 4 0 4]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data (k=7).\n');
            
            A = [mvnrnd([3 1],[0.5 0; 0 0.5],100);
                 mvnrnd([3 3],[0.5 0; 0 0.5],100);
                 mvnrnd([1 3],[0.5 0; 0 0.5],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1','2','3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = knn_classifier(s_tr,7);
            
            [labels_idx_hat,~,~,~] = cl.classify(s_ts);
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                scatter(s_tr.samples(s_tr.labels_idx == 1,1),s_tr.samples(s_tr.labels_idx == 1,2),'x','r');
                scatter(s_tr.samples(s_tr.labels_idx == 2,1),s_tr.samples(s_tr.labels_idx == 2,2),'x','g');
                scatter(s_tr.samples(s_tr.labels_idx == 3,1),s_tr.samples(s_tr.labels_idx == 3,2),'x','b');
                scatter(s_ts.samples(labels_idx_hat == 1,1),s_ts.samples(labels_idx_hat == 1,2),'o','r');
                scatter(s_ts.samples(labels_idx_hat == 2,1),s_ts.samples(labels_idx_hat == 2,2),'o','g');
                scatter(s_ts.samples(labels_idx_hat == 3,1),s_ts.samples(labels_idx_hat == 3,2),'o','b');
                ptmp = allcomb(0:0.1:4,0:0.1:4);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([0 4 0 4]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
