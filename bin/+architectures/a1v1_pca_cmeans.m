classdef a1v1_pca_cmeans < architecture
    properties (GetAccess=public,SetAccess=immutable)
        pca_kept_energy;
    end
    
    methods (Access=public)
        function [obj] = a1v1_pca_cmeans(train_dataset_plain,pca_kept_energy)
            assert(tc.scalar(train_dataset_plain) && tc.dataset(train_dataset_plain));
            assert(tc.scalar(pca_kept_energy) && tc.unitreal(pca_kept_energy));
            
            t_pca = transforms.pca(train_dataset_plain,pca_kept_energy);
            train_dataset_plain_1 = t_pca.code(train_dataset_plain);
            
            classifier = classifiers.cmeans(train_dataset_plain_1);
            
            obj = obj@architecture(train_dataset_plain.subsamples(1),{t_pca},classifier);
            obj.pca_kept_energy = pca_kept_energy;
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "architecture.a1v1_pca_cmeans".\n');
        
            fprintf('  Proper construction.\n');
            
            A = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)];
            c = [1*ones(100,1);2*ones(100,1)];
            s = dataset({'1' '2'},A,c);
            
            ar = architectures.a1v1_pca_cmeans(s,0.90);
            
            assert(ar.pca_kept_energy == 0.90);
            assert(length(ar.one_sample_plain.classes) == 2);
            assert(strcmp(ar.one_sample_plain.classes{1},'1'));
            assert(strcmp(ar.one_sample_plain.classes{2},'2'));
            assert(ar.one_sample_plain.classes_count == 2);
            assert(tc.check(ar.one_sample_plain.samples == A(1,:)));
            assert(tc.check(ar.one_sample_plain.labels_idx == c(1)));
            assert(ar.one_sample_plain.samples_count == 1);
            assert(ar.one_sample_plain.features_count == 2);
            assert(ar.one_sample_plain.compatible(s));
            assert(tc.vector(ar.transforms) && tc.cell(ar.transforms) && ...
                   tc.check(cellfun(@(c)tc.scalar(c) && tc.transform(c),ar.transforms)));
            assert(tc.scalar(ar.classifier) && tc.classifier(ar.classifier));
            assert(ar.transforms{1}.one_sample_plain.compatible(s));
            assert(ar.transforms{1}.one_sample_coded.compatible(ar.classifier.one_sample));
            assert(ar.classifier.one_sample.compatible(ar.transforms{1}.one_sample_coded));
        
            clearvars -except display;
        
            fprintf('  Function classify.\n');
            
            fprintf('    With clearly separated data.\n');
            
            A = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)];
            c = [1*ones(100,1);2*ones(100,1)];            
            s = dataset({'1' '2'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            ar = architectures.a1v1_pca_cmeans(s_tr,0.90);
            [labels_idx_hat,~,labels_idx_hat2,~,...
                score,conf_matrix,misclassified] = ar.classify(s_ts);
            
            assert(tc.check(labels_idx_hat == s_ts.labels_idx));
            assert(tc.check(labels_idx_hat2 == [repmat([1 2],20,1);repmat([2 1],20,1)]));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = ar.classify(dataset({'1' '2'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);
                hold off;
                title('Original data space decision surface.');
                subplot(1,2,2);
                hold on;
                s_p_tr = ar.transforms{1}.code(s_tr);
                s_p_ts = ar.transforms{1}.code(s_ts);
                gscatter(s_p_tr.samples,zeros(size(s_p_tr.samples)),s_p_tr.labels_idx,'rg','o',6);
                gscatter(s_p_ts.samples,zeros(size(s_p_ts.samples)),s_p_ts.labels_idx,'rg','o',6);
                ptmp = (-3:0.05:3)';
                l = ar.classifier.classify(dataset({'1' '2'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp,zeros(size(ptmp)),l,'rg','*',2);
                axis([-3 3 -3 3]);
                hold off;
                title('PCA transformed data space decision surface.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With mostly clearly separated data.\n');
            
            A_tr = [mvnrnd([3 1],[0.01 0; 0 0.01],80);
                    mvnrnd([1 3],[0.01 0; 0 0.01],80)];
            A_ts = [mvnrnd([3 1],[0.01 0; 0 0.01],19);
                    1 3;
                    mvnrnd([1 3],[0.01 0; 0 0.01],19)
                    3 1];
            c_tr = [1*ones(80,1);2*ones(80,1)];
            c_ts = [1*ones(20,1);2*ones(20,1)];
            s_tr = dataset({'1' '2'},A_tr,c_tr);
            s_ts = dataset({'1' '2'},A_ts,c_ts);  
            
            ar = architectures.a1v1_pca_cmeans(s_tr,0.90);
            [labels_idx_hat,~,~,~,...
                score,conf_matrix,misclassified] = ar.classify(s_ts);
            
            assert(tc.check(labels_idx_hat(1:19) == s_ts.labels_idx(1:19)));
            assert(tc.check(labels_idx_hat(21:39) == s_ts.labels_idx(21:39)));
            assert(labels_idx_hat(20) == 2);
            assert(labels_idx_hat(40) == 1);
            assert(score == 95);
            assert(tc.check(conf_matrix == [19 1; 1 19]));
            assert(tc.check(misclassified == [20 40]'));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = ar.classify(dataset({'1' '2'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);
                hold off;
                title('Original data space decision surface.');
                subplot(1,2,2);
                hold on;
                s_p_tr = ar.transforms{1}.code(s_tr);
                s_p_ts = ar.transforms{1}.code(s_ts);
                gscatter(s_p_tr.samples,zeros(size(s_p_tr.samples)),s_p_tr.labels_idx,'rg','o',6);
                gscatter(s_p_ts.samples,zeros(size(s_p_ts.samples)),s_p_ts.labels_idx,'rg','o',6);
                ptmp = (-3:0.05:3)';
                l = ar.classifier.classify(dataset({'1' '2'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp,zeros(size(ptmp)),l,'rg','*',2);
                axis([-3 3 -3 3]);
                hold off;
                title('PCA transformed data space decision surface.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Without clearly separated data.\n');
            
            A = [mvnrnd([3 1],[0.2 0; 0 1],100);
                 mvnrnd([1 3],[0.2 0; 0 1],100)];
            c = [1*ones(100,1);2*ones(100,1)];            
            s = dataset({'1' '2'},A,c);
            [tr_i,ts_i] = s.partition('holdout',0.2);            
            s_tr = s.subsamples(tr_i);
            s_ts = s.subsamples(ts_i);
            
            ar = architectures.a1v1_pca_cmeans(s_tr,0.90);
            
            if exist('display','var') && (display == true)
                figure();
                hold on;
                gscatter(s_tr.samples(:,1),s_tr.samples(:,2),s_tr.labels_idx,'rg','o',6);
                gscatter(s_ts.samples(:,1),s_ts.samples(:,2),s_ts.labels_idx,'rg','o',6);
                ptmp = allcomb(-1:0.05:5,-1:0.05:5);
                l = ar.classify(dataset({'1' '2'},ptmp,ones(length(ptmp),1)));
                gscatter(ptmp(:,1),ptmp(:,2),l,'rg','*',2);
                axis([-1 5 -1 5]);
                hold off;
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
