classdef classifier_info
    properties (GetAccess=public,SetAccess=immutable)
        labels;
        labels_count;
        labels_idx;
    end
    
    methods (Access=public)
        function [obj] = classifier_info(labels,labels_idx)
            assert(check.vector(labels));
            assert(check.cell(labels));
            assert(check.checkf(@check.scalar,labels));
            assert(check.checkf(@check.string,labels));
            assert(check.vector(labels_idx));
            assert(check.natural(labels_idx));
            assert(check.checkv(labels_idx >= 1 & labels_idx <= length(labels)));
            
            obj.labels = utils.common.force_row(labels);
            obj.labels_count = length(labels);
            obj.labels_idx = utils.common.force_row(labels_idx);
        end
        
        function [o] = compatible(obj,samples)
            assert(check.scalar(obj));
            assert(check.classifier_info(obj));
            assert(check.dataset(samples));
            
            N = dataset.count(samples);
            
            o = length(obj.labels_idx) == N;
        end
        
        function [tr_index,ts_index] = partition(obj,param)
            assert(check.scalar(obj));
            assert(check.classifier_info(obj));
            assert(check.scalar(param));
            assert(check.unitreal(param));
            assert(param > 0 && param < 1);
               
            partition = cvpartition(obj.labels_idx,'holdout',param);
            
            tr_index = training(partition,1)';
            ts_index = test(partition,1)';
        end
        
        function [new_classifier_info] = subsample(obj,index)
            assert(check.scalar(obj));
            assert(check.classifier_info(obj));
            assert(check.vector(index));
            assert((check.logical(index) && check.match_dims(obj.labels_idx,index)) || ...
                   (check.natural(index) && check.checkv(index >= 1 & index <= size(obj.labels_idx,2))));
                   
            new_classifier_info = classifier_info(obj.labels,obj.labels_idx(index));
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "classifier_info".\n');
            
            fprintf('  Proper construction.\n');
            
            ci = classifier_info({'1' '2'},[1*ones(1,10) 2*ones(1,10)]);
            
            assert(check.same(ci.labels,{'1' '2'}));
            assert(ci.labels_count == 2);
            assert(check.same(ci.labels_idx,[1*ones(1,10) 2*ones(1,10)]));
            
            clearvars -except test_figure;
            
            fprintf('  Function "compatible".\n');
            
            ci = classifier_info({'1' '2'},[1*ones(1,10) 2*ones(1,10)]);
            
            assert(ci.compatible(rand(2,20)) == true);
            assert(ci.compatible(rand(40,20)) == true);
            assert(ci.compatible(rand(2,19)) == false);
            assert(ci.compatible(rand(2,21)) == false);
            assert(ci.compatible(rand(20,2)) == false);
            assert(ci.compatible(rand(8,8,1,20)) == true);
            assert(ci.compatible(rand(16,16,3,20)) == true);
            assert(ci.compatible(rand(20,8,3,19)) == false);
            assert(ci.compatible(rand(8,20,1,21)) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "partition".\n');
            
            ci = classifier_info({'1' '2' '3'},[1 2 3 1 2 3 1 2 3 1 2 3]);

            [tr,ts] = ci.partition(0.33);
            
            assert(sum(tr == 0) == 3);
            assert(sum(tr == 1) == 9);
            assert(sum(ts == 0) == 9);
            assert(sum(ts == 1) == 3);

            clearvars -except test_figure;
            
            fprintf('  Function "subsample".\n');
            
            ci = classifier_info({'1' '2' '3'},[1 2 3 1 2 3 1 2 3 1 2 3]);
            
            ci_1 = ci.subsample(1);
            ci_2 = ci.subsample([1 2 5]);
            ci_3 = ci.subsample([true true true false false false true false true false true false]);

            assert(check.same(ci_1.labels,ci.labels));
            assert(ci_1.labels_count == 3);
            assert(check.same(ci_1.labels_idx,1));
            assert(check.same(ci_2.labels,ci.labels));
            assert(ci_2.labels_count == 3);
            assert(check.same(ci_2.labels_idx,[1 2 2]));
            assert(check.same(ci_3.labels,ci.labels));
            assert(ci_3.labels_count == 3);
            assert(check.same(ci_3.labels_idx,[1 2 3 1 3 2]));
            
            clearvars -except test_figure;
        end
    end
end

