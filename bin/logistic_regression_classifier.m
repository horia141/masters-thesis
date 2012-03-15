classdef logistic_regression_classifier < classifier
    properties (GetAccess=public,SetAccess=immutable)
        sep_plane_coeffs;
    end
    
    methods (Access=public)
        function [obj] = logistic_regression_classifier(train_samples)
            assert(tc.scalar(train_samples) && tc.samples_set(train_samples));
            assert(train_samples.samples_count > 0);
            
            obj = obj@classifier(train_samples);
            obj.sep_plane_coeffs = mnrfit(train_samples.samples,train_samples.labels_idx);
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = do_classify(obj,samples)
            classes_probs = mnrval(obj.sep_plane_coeffs,samples.samples);
            [classes_probs_sorted,classes_probs_index] = sort(classes_probs,2,'descend');
            
            labels_idx_hat = classes_probs_index(:,1);
            labels_confidence = classes_probs_sorted(:,1);
            labels_idx_hat2 = classes_probs_index;
            labels_confidence2 = classes_probs_sorted;
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "logistic_regression_classifier".\n');
            
            fprintf('  Proper construction.\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([3 3],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1' '2' '3'},A,c);
            cl = logistic_regression_classifier(s);
            
            assert(utils.approx(cl.sep_plane_coeffs,mnrfit(A,c)));
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    With clearly separated data.\n');
            
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([3 3],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);2*ones(100,1);3*ones(100,1)];
            
            s = samples_set({'1' '2' '3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = logistic_regression_classifier(s_tr);
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                score,conf_matrix,misclassified] = cl.classify(s_ts);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(utils.approx(labels_confidence,ones(60,1),1e-3));
            assert(tc.matrix(labels_idx_hat2));
            assert(tc.natural(labels_idx_hat2));
            assert(tc.check(labels_idx_hat2 >= 1 & labels_idx_hat2 <= 3));
            assert(utils.approx(labels_confidence2,[ones(60,1),zeros(60,2)],1e-3));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
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
            
            s_tr = samples_set({'1' '2' '3'},A_tr,c_tr);
            s_ts = samples_set({'1' '2' '3'},A_ts,c_ts);
            
            cl = logistic_regression_classifier(s_tr);
            
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
            assert(utils.approx(labels_confidence,ones(60,1),1e-3));
            assert(tc.matrix(labels_idx_hat2));
            assert(tc.natural(labels_idx_hat2));
            assert(tc.check(labels_idx_hat2 >= 1 & labels_idx_hat2 <= 3));
            assert(utils.approx(labels_confidence2,[ones(60,1),zeros(60,2)],1e-3));
            assert(score == 90);
            assert(tc.check(conf_matrix == [18 1 1; 1 18 1; 1 1 18]));
            assert(tc.check(misclassified == [19 20 39 40 59 60]'));
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
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
            
            s = samples_set({'1' '2' '3'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);
            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            cl = logistic_regression_classifier(s_tr);
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rgb','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rgb','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = cl.classify(samples_set({'1' '2' '3'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rgb','*',2);
                axis([-1 5 -1 5]);                
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
