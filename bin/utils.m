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
        
        function [o] = approx(v1,v2,epsilon)
            assert(tc.number(v1));
            assert(tc.number(v2));
            assert(~exist('epsilon','var') || tc.unitreal(epsilon));
            
            if exist('epsilon','var')
                epsilon_t = epsilon;
            else
                epsilon_t = 1e-6;
            end
            
            r = abs(v1 - v2) < epsilon_t;
            o = all(r(:));
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
            
            fprintf('  Testing function "approx".\n');
            
            t = rand(100,100);
            
            assert(utils.approx(1,1) == true);
            assert(utils.approx(1,1 + 1e-7) == true);
            assert(utils.approx(t,sqrt(t .^ 2)) == true);
            assert(utils.approx(t,(t - repmat(mean(t,1),100,1)) + repmat(mean(t,1),100,1)) == true);
            assert(utils.approx(1,1.5,0.9) == true);
            assert(utils.approx(2,2.5) == false);
            assert(utils.approx(1,1 + 1e-7,1e-9) == false);
            
            clear all;
        end
    end
end
