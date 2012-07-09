classdef regressor_info
    properties (GetAccess=public,SetAccess=immutable)
        target_values;
    end
    
    methods (Access=public)
        function [obj] = regressor_info(target_values)
            assert(check.vector(target_values));
            assert(check.number(target_values));
            
            obj.target_values = target_values;
        end
        
        function [o] = compatible(obj,samples)
            assert(check.scalar(obj));
            assert(check.regressor_info(obj));
            assert(check.dataset(samples));
            
            N = dataset.count(samples);
            
            o = length(obj.target_values) == N;
        end
        
        function [tr_index,ts_index] = partition(obj,param)
            assert(check.scalar(obj));
            assert(check.regressor_info(obj));
            assert(check.scalar(param));
            assert(check.unitreal(param));
            assert(param > 0 && param < 1);
               
            partition = cvpartition(obj.target_values,'holdout',param);
            
            tr_index = training(partition,1)';
            ts_index = test(partition,1)';
        end
        
        function [new_regressor_info] = subsample(obj,index)
            assert(check.scalar(obj));
            assert(check.regressor_info(obj));
            assert(check.vector(index));
            assert((check.logical(index) && check.match_dims(obj.target_values,index)) || ...
                   (check.natural(index) && check.checkv(index >= 1 & index <= size(obj.target_values,2))));
                   
            new_regressor_info = regressor_info(obj.target_values(index));
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "regressor_info".\n');
            
            fprintf('  Proper construction.\n');
            
            ri = regressor_info([1*ones(1,10) 2*ones(1,10)]);
            
            assert(check.same(ri.target_values,[1*ones(1,10) 2*ones(1,10)]));
            
            clearvars -except test_figure;
            
            fprintf('  Function "compatible".\n');
            
            ri = regressor_info([1*ones(1,10) 2*ones(1,10)]);
            
            assert(ri.compatible(rand(2,20)) == true);
            assert(ri.compatible(rand(40,20)) == true);
            assert(ri.compatible(rand(2,19)) == false);
            assert(ri.compatible(rand(2,21)) == false);
            assert(ri.compatible(rand(20,2)) == false);
            assert(ri.compatible(rand(8,8,1,20)) == true);
            assert(ri.compatible(rand(16,16,3,20)) == true);
            assert(ri.compatible(rand(20,8,3,19)) == false);
            assert(ri.compatible(rand(8,20,1,21)) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "partition".\n');
            
            ri = regressor_info([1 2 3 1 2 3 1 2 3 1 2 3]);

            [tr,ts] = ri.partition(0.33);
            
            assert(sum(tr == 0) == 3);
            assert(sum(tr == 1) == 9);
            assert(sum(ts == 0) == 9);
            assert(sum(ts == 1) == 3);

            clearvars -except test_figure;
            
            fprintf('  Function "subsample".\n');
            
            ri = regressor_info([1 2 3 1 2 3 1 2 3 1 2 3]);
            
            ri_1 = ri.subsample(1);
            ri_2 = ri.subsample([1 2 5]);
            ri_3 = ri.subsample([true true true false false false true false true false true false]);

            assert(check.same(ri_1.target_values,1));
            assert(check.same(ri_2.target_values,[1 2 2]));
            assert(check.same(ri_3.target_values,[1 2 3 1 3 2]));
            
            clearvars -except test_figure;
        end
    end
end

