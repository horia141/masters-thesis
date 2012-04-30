classdef classification_info
    properties (GetAccess=public,SetAccess=immutable)
        labels;
        labels_count;
        labels_idx;
    end
    
    methods (Access=public)
        function [obj] = classification_info(labels,labels_idx)
            assert(tc.vector(labels));
            assert(tc.labels(labels));
            assert(tc.vector(labels_idx));
            assert(tc.labels_idx(labels_idx,labels));
            
            obj.labels = utils.force_col(labels);
            obj.labels_count = length(labels);
            obj.labels_idx = utils.force_col(labels_idx);
        end
        
        function [o] = compatible(obj,samples)
            assert(tc.scalar(obj));
            assert(tc.classification_info(obj));
            assert(tc.dataset(samples));
            
            N = dataset.count(samples);
            
            o = length(obj.labels_idx) == N;
        end
        
        function [tr_index,ts_index] = partition(obj,type,param)
            assert(tc.scalar(obj));
            assert(tc.classification_info(obj));
            assert(tc.scalar(type));
            assert(tc.string(type));
            assert(tc.one_of(type,'kfold','holdout'));
            assert(tc.scalar(param));
            assert(tc.number(param));
            assert((strcmp(type,'kfold') && tc.natural(param) && (param >= 2)) || ...
                   (strcmp(type,'holdout') && tc.unitreal(param)));
               
            partition = cvpartition(obj.labels_idx,type,param);
            
            tr_index = false(length(obj.labels_idx),partition.NumTestSets);
            ts_index = false(length(obj.labels_idx),partition.NumTestSets);
            
            for ii = 1:partition.NumTestSets
                tr_index(:,ii) = training(partition,ii)';
                ts_index(:,ii) = test(partition,ii)';
            end
        end
        
        function [new_classification_info] = subsample(obj,index)
            assert(tc.scalar(obj));
            assert(tc.classification_info(obj));
            assert(tc.vector(index));
            assert((tc.logical(index) && tc.match_dims(obj.labels_idx,index)) || ...
                   (tc.natural(index) && tc.check(index >= 1 & index <= size(obj.labels_idx,1))));
                   
            new_classification_info = classification_info(obj.labels,obj.labels_idx(index));
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "classification_info".\n');
            
            fprintf('  Proper construction.\n');
            
            ci = classification_info({'1' '2'},[1*ones(1,10) 2*ones(1,10)]);
            
            assert(tc.same(ci.labels,{'1';'2'}));
            assert(ci.labels_count == 2);
            assert(tc.same(ci.labels_idx,[1*ones(10,1);2*ones(10,1)]));
            
            clearvars -except display;
            
            fprintf('  Function "compatible".\n');
            
            ci = classification_info({'1' '2'},[1*ones(1,10) 2*ones(1,10)]);
            
            assert(ci.compatible(rand(20,2)) == true);
            assert(ci.compatible(rand(20,40)) == true);
            assert(ci.compatible(rand(19,2)) == false);
            assert(ci.compatible(rand(21,2)) == false);
            assert(ci.compatible(rand(2,20)) == false);
            assert(ci.compatible(rand(8,8,1,20)) == true);
            assert(ci.compatible(rand(16,16,3,20)) == true);
            assert(ci.compatible(rand(20,8,3,19)) == false);
            assert(ci.compatible(rand(8,20,1,21)) == false);
            
            fprintf('  Function "partition".\n');
            
            fprintf('    2-fold partition.\n');
            
            ci = classification_info({'1' '2' '3'},[1 2 3 1 2 3 1 2 3 1 2 3]);

            [tr,ts] = ci.partition('kfold',2);

            assert(sum(tr(:,1) == 0) == 6);
            assert(sum(tr(:,1) == 1) == 6);
            assert(sum(tr(:,2) == 0) == 6);
            assert(sum(tr(:,2) == 1) == 6);
            assert(sum(ts(:,1) == 0) == 6);
            assert(sum(ts(:,1) == 1) == 6);
            assert(sum(ts(:,2) == 0) == 6);
            assert(sum(ts(:,2) == 1) == 6);

            clearvars -except display;

            fprintf('    Holdout partition with p=0.33.\n');
            
            ci = classification_info({'1' '2' '3'},[1 2 3 1 2 3 1 2 3 1 2 3]);

            [tr,ts] = ci.partition('holdout',0.33);
            
            assert(sum(tr == 0) == 3);
            assert(sum(tr == 1) == 9);
            assert(sum(ts == 0) == 9);
            assert(sum(ts == 1) == 3);

            clearvars -except display;
            
            fprintf('  Function "sublabels".\n');
            
            ci = classification_info({'1' '2' '3'},[1 2 3 1 2 3 1 2 3 1 2 3]);
            
            ci_1 = ci.subsample(1);
            ci_2 = ci.subsample([1 2 5]);
            ci_3 = ci.subsample([true true true false false false true false true false true false]);
            
            assert(tc.same(ci_1.labels,ci.labels));
            assert(ci_1.labels_count == 3);
            assert(tc.same(ci_1.labels_idx,1));
            assert(tc.same(ci_2.labels,ci.labels));
            assert(ci_2.labels_count == 3);
            assert(tc.same(ci_2.labels_idx,[1 2 2]'));
            assert(tc.same(ci_3.labels,ci.labels));
            assert(ci_3.labels_count == 3);
            assert(tc.same(ci_3.labels_idx,[1 2 3 1 3 2]'));
        end
    end
end

