classdef level
    enumeration
        Error;
        Status;
        Details;
    end
    
    methods (Access=private)
        function [code] = num_code(obj)
            switch (obj)
                case logging.level.Error
                    code = 1;
                case logging.level.Status
                    code = 2;
                case logging.level.Details
                    code = 3;
            end
        end
    end
    
    methods (Access=public)
        function [o] = ge(obj,other_obj)
            assert(tc.scalar(other_obj) && tc.logging_level(other_obj));
            
            persistent p_level_ge;
            
            if tc.empty(p_level_ge)
                p_level_ge = [true  true  true;
                              false true  true;
                              false false true];
            end
            
            obj_code = obj.num_code();
            other_obj_code = other_obj.num_code();
            
            o = p_level_ge(obj_code,other_obj_code);
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logging.level".\n');
            
            fprintf('  Proper construction.\n');
            
            a = logging.level.Error;
            b = logging.level.Status;
            c = logging.level.Details;
            
            assert(a.num_code() == 1);
            assert(b.num_code() == 2);
            assert(c.num_code() == 3);
            
            clearvars -except display;
            
            fprintf('  Functions "eq" and "ne".\n');
            
            a = logging.level.Error;
            b = logging.level.Status;
            c = logging.level.Details;
            ap = logging.level.Error;
            bp = logging.level.Status;
            cp = logging.level.Details;
            
            assert(a == a);
            assert(a ~= b);
            assert(a ~= c);
            assert(a == ap);
            assert(b ~= a);
            assert(b == b);
            assert(b ~= c);
            assert(b == bp);
            assert(c ~= a);
            assert(c ~= b);
            assert(c == c);
            assert(c == cp);
            
            clearvars -except display;
            
            fprintf('  Function "char".\n');
            
            a = logging.level.Error;
            b = logging.level.Status;
            c = logging.level.Details;
            
            assert(strcmp(a.char(),'Error'));
            assert(strcmp(b.char(),'Status'));
            assert(strcmp(c.char(),'Details'));
            
            clearvars -except display;
            
            fprintf('  Function "ge".\n');
            
            a = logging.level.Error;
            b = logging.level.Status;
            c = logging.level.Details;
            
            assert(a >= a);
            assert(a >= b);
            assert(a >= c);
            assert(~(b >= a));
            assert(b >= b);
            assert(b >= c);
            assert(~(c >= a));
            assert(~(c >= b));
            assert(c >= c);
            
            clearvars -except display;
        end
    end
end