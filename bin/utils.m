classdef utils
    methods (Static,Access=public)
        function [o_v] = force_row(v)
            assert(tc.vector(v));
            
            if size(v,2) == length(v)
                o_v = v;
            else
                o_v = v';
            end
        end
        
        function [o_v] = force_col(v)
            assert(tc.vector(v));
            
            if size(v,1) == length(v)
                o_v = v;
            else
                o_v = v';
            end
        end
    end
    
    methods (Static,Access=public)
        function test
            fprintf('Testing "utils".\n');
            
            fprintf('  Testing function "force_row".\n');
            
            assert(all(utils.force_row([1 2 3]) == [1 2 3]));
            assert(all(utils.force_row([1;2;3]) == [1 2 3]));
            assert(all(utils.force_row(zeros(1,45)) == zeros(1,45)));
            assert(all(utils.force_row(ones(41,1)) == ones(1,41)));
            
            fprintf('  Testing function "force_col".\n');
            
            assert(all(utils.force_col([1 2 3]) == [1;2;3]));
            assert(all(utils.force_col([1;2;3]) == [1;2;3]));
            assert(all(utils.force_col(zeros(1,45)) == zeros(45,1)));
            assert(all(utils.force_col(ones(41,1)) == ones(41,1)));
        end
    end
end
