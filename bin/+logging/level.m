classdef level
    enumeration
        Experiment;
        Classifier;
        Regressor;
        Transform;
        Results;
    end
    
    methods (Access=public)
        function [o] = ge(obj,other_obj)
            assert(check.scalar(obj));
            assert(check.logging_level(obj));
            assert(check.scalar(other_obj))
            assert(check.logging_level(other_obj));
            
            persistent p_level_ge;
            
            if check.empty(p_level_ge)
                p_level_ge = [true  true  true  true  true ;
                              false true  false false false;
                              false false true  false false;
                              false false false true  false;
                              true  true  true  true  true ;
                              false false false false false];
            end

            obj_code = obj.num_code();
            other_obj_code = other_obj.num_code();

            o = p_level_ge(obj_code,other_obj_code);
        end
        
        function [o] = le(obj,other_obj)
            assert(check.scalar(obj));
            assert(check.logging_level(obj));
            assert(check.scalar(other_obj))
            assert(check.logging_level(other_obj));
            
            o = ~(obj >= other_obj);
        end
    end
    
    methods (Access=private)
        function [code] = num_code(obj)
            switch obj
                case logging.level.Experiment
                    code = 1;
                case logging.level.Classifier
                    code = 2;
                case logging.level.Regressor
                    code = 3;
                case logging.level.Transform
                    code = 4;
                case logging.level.Results
                    code = 5;
            end
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logging.level".\n');
            
            fprintf('  Proper construction.\n');
            
            a = logging.level.Experiment;
            b = logging.level.Classifier;
            c = logging.level.Regressor;
            d = logging.level.Transform;
            e = logging.level.Results;
            
            assert(a.num_code() == 1);
            assert(b.num_code() == 2);
            assert(c.num_code() == 3);
            assert(d.num_code() == 4);
            assert(e.num_code() == 5);
            
            clearvars -except test_figure;
            
            fprintf('  Functions "eq" and "ne".\n');
            
            a = logging.level.Experiment;
            b = logging.level.Classifier;
            c = logging.level.Regressor;
            d = logging.level.Transform;
            e = logging.level.Results;
            ap = logging.level.Experiment;
            bp = logging.level.Classifier;
            cp = logging.level.Regressor;
            dp = logging.level.Transform;
            ep = logging.level.Results;
            
            assert(a == a);
            assert(a ~= b);
            assert(a ~= c);
            assert(a ~= d);
            assert(a ~= e);
            assert(a == ap);
            assert(b ~= a);
            assert(b == b);
            assert(b ~= c);
            assert(b ~= d);
            assert(b ~= e);
            assert(b == bp);
            assert(c ~= a);
            assert(c ~= b);
            assert(c == c);
            assert(c ~= d);
            assert(c ~= e);
            assert(c == cp);
            assert(d ~= a);
            assert(d ~= b);
            assert(d ~= c);
            assert(d == d);
            assert(d ~= e);
            assert(d == dp);
            assert(e ~= a);
            assert(e ~= b);
            assert(e ~= c);
            assert(e ~= d);
            assert(e == e);
            assert(e == ep);
            
            clearvars -except test_figure;
            
            fprintf('  Function "char".\n');
            
            a = logging.level.Experiment;
            b = logging.level.Classifier;
            c = logging.level.Regressor;
            d = logging.level.Transform;
            e = logging.level.Results;
            
            assert(strcmp(a.char(),'Experiment'));
            assert(strcmp(b.char(),'Classifier'));
            assert(strcmp(c.char(),'Regressor'));
            assert(strcmp(d.char(),'Transform'));
            assert(strcmp(e.char(),'Results'));
            
            clearvars -except test_figure;
            
            fprintf('  Functions "ge" and "le".\n');
            
            a = logging.level.Experiment;
            b = logging.level.Classifier;
            c = logging.level.Regressor;
            d = logging.level.Transform;
            e = logging.level.Results;
            
            assert(a >= a);
            assert(a >= b);
            assert(a >= c);
            assert(a >= d);
            assert(a >= e);
            assert(b <= a);
            assert(b >= b);
            assert(b <= c);
            assert(b <= d);
            assert(b <= e);
            assert(c <= a);
            assert(c <= b);
            assert(c >= c);
            assert(c <= d);
            assert(c <= e);
            assert(d <= a);
            assert(d <= b);
            assert(d <= c);
            assert(d >= d);
            assert(d <= e);
            assert(e >= a);
            assert(e >= b);
            assert(e >= c);
            assert(e >= d);
            assert(e >= e);
            
            clearvars -except test_figure;
        end
    end
end
