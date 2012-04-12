classdef level
    enumeration
        TopLevel;
        Experiment;
        Architecture;
        Transform;
        Classifier;
        Dataset_IO;
        Results;
        Error;
        All;
    end
    
    methods (Access=public)
        function [o] = ge(obj,other_obj)
            assert(tc.scalar(obj));
            assert(tc.logging_level(obj));
            assert(tc.scalar(other_obj))
            assert(tc.logging_level(other_obj));
            
            persistent p_level_ge;
            
            if tc.empty(p_level_ge)
                p_level_ge = [true  true  true  true  true  true  false false true;
                              false true  true  true  true  true  false false true;
                              false false true  true  true  true  false false true;
                              false false false true  false false false false true;
                              false false false false true  false false false true;
                              false false false false false true  false false true;
                              true  true  true  true  true  true  true  false true;
                              true  true  true  true  true  true  false true  true;
                              false false false false false false false false true];
            end
            
            obj_code = obj.num_code();
            other_obj_code = other_obj.num_code();
            
            o = p_level_ge(obj_code,other_obj_code);
        end
        
        function [o] = le(obj,other_obj)
            assert(tc.scalar(obj));
            assert(tc.logging_level(obj));
            assert(tc.scalar(other_obj))
            assert(tc.logging_level(other_obj));
            
            o = ~(obj >= other_obj);
        end
    end
    
    methods (Access=private)
        function [code] = num_code(obj)
            switch obj
                case logging.level.TopLevel
                    code = 1;
                case logging.level.Experiment
                    code = 2;
                case logging.level.Architecture
                    code = 3;
                case logging.level.Transform
                    code = 4;
                case logging.level.Classifier
                    code = 5;
                case logging.level.Dataset_IO
                    code = 6;
                case logging.level.Results
                    code = 7;
                case logging.level.Error
                    code = 8;
                case logging.level.All
                    code = 9;
            end
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logging.level".\n');
            
            fprintf('  Proper construction.\n');
            
            a = logging.level.TopLevel;
            b = logging.level.Experiment;
            c = logging.level.Architecture;
            d = logging.level.Transform;
            e = logging.level.Classifier;
            f = logging.level.Dataset_IO;
            g = logging.level.Results;
            h = logging.level.Error;
            i = logging.level.All;
            
            assert(a.num_code() == 1);
            assert(b.num_code() == 2);
            assert(c.num_code() == 3);
            assert(d.num_code() == 4);
            assert(e.num_code() == 5);
            assert(f.num_code() == 6);
            assert(g.num_code() == 7);
            assert(h.num_code() == 8);
            assert(i.num_code() == 9);
            
            clearvars -except display;
            
            fprintf('  Functions "eq" and "ne".\n');
            
            a = logging.level.TopLevel;
            b = logging.level.Experiment;
            c = logging.level.Architecture;
            d = logging.level.Transform;
            e = logging.level.Classifier;
            f = logging.level.Dataset_IO;
            g = logging.level.Results;
            h = logging.level.Error;
            i = logging.level.All;
            ap = logging.level.TopLevel;
            bp = logging.level.Experiment;
            cp = logging.level.Architecture;
            dp = logging.level.Transform;
            ep = logging.level.Classifier;
            fp = logging.level.Dataset_IO;
            gp = logging.level.Results;
            hp = logging.level.Error;
            ip = logging.level.All;
            
            assert(a == a);
            assert(a ~= b);
            assert(a ~= c);
            assert(a ~= d);
            assert(a ~= e);
            assert(a ~= f);
            assert(a ~= g);
            assert(a ~= h);
            assert(a ~= i);
            assert(a == ap);
            assert(b ~= a);
            assert(b == b);
            assert(b ~= c);
            assert(b ~= d);
            assert(b ~= e);
            assert(b ~= f);
            assert(b ~= g);
            assert(b ~= h);
            assert(b ~= i);
            assert(b == bp);
            assert(c ~= a);
            assert(c ~= b);
            assert(c == c);
            assert(c ~= d);
            assert(c ~= e);
            assert(c ~= f);
            assert(c ~= g);
            assert(c ~= h);
            assert(c ~= i);
            assert(c == cp);
            assert(d ~= a);
            assert(d ~= b);
            assert(d ~= c);
            assert(d == d);
            assert(d ~= e);
            assert(d ~= f);
            assert(d ~= g);
            assert(d ~= h);
            assert(d ~= i);
            assert(d == dp);
            assert(e ~= a);
            assert(e ~= b);
            assert(e ~= c);
            assert(e ~= d);
            assert(e == e);
            assert(e ~= f);
            assert(e ~= g);
            assert(e ~= h);
            assert(e ~= i);
            assert(e == ep);
            assert(f ~= a);
            assert(f ~= b);
            assert(f ~= c);
            assert(f ~= d);
            assert(f ~= e);
            assert(f == f);
            assert(f ~= g);
            assert(f ~= h);
            assert(f ~= i);
            assert(f == fp);
            assert(g ~= a);
            assert(g ~= b);
            assert(g ~= c);
            assert(g ~= d);
            assert(g ~= e);
            assert(g ~= f);
            assert(g == g);
            assert(g ~= h);
            assert(g ~= i);
            assert(g == gp);
            assert(h ~= a);
            assert(h ~= b);
            assert(h ~= c);
            assert(h ~= d);
            assert(h ~= e);
            assert(h ~= f);
            assert(h ~= g);
            assert(h == h);
            assert(h ~= i);
            assert(h == hp);
            assert(i ~= a);
            assert(i ~= b);
            assert(i ~= c);
            assert(i ~= d);
            assert(i ~= e);
            assert(i ~= f);
            assert(i ~= g);
            assert(i ~= h);
            assert(i == i);
            assert(i == ip);
            
            clearvars -except display;
            
            fprintf('  Function "char".\n');
            
            a = logging.level.TopLevel;
            b = logging.level.Experiment;
            c = logging.level.Architecture;
            d = logging.level.Transform;
            e = logging.level.Classifier;
            f = logging.level.Dataset_IO;
            g = logging.level.Results;
            h = logging.level.Error;
            i = logging.level.All;
            
            assert(strcmp(a.char(),'TopLevel'));
            assert(strcmp(b.char(),'Experiment'));
            assert(strcmp(c.char(),'Architecture'));
            assert(strcmp(d.char(),'Transform'));
            assert(strcmp(e.char(),'Classifier'));
            assert(strcmp(f.char(),'Dataset_IO'));
            assert(strcmp(g.char(),'Results'));
            assert(strcmp(h.char(),'Error'));
            assert(strcmp(i.char(),'All'));
            
            clearvars -except display;
            
            fprintf('  Functions "ge" and "le".\n');
            
            a = logging.level.TopLevel;
            b = logging.level.Experiment;
            c = logging.level.Architecture;
            d = logging.level.Transform;
            e = logging.level.Classifier;
            f = logging.level.Dataset_IO;
            g = logging.level.Results;
            h = logging.level.Error;
            i = logging.level.All;
            
            assert(a >= a);
            assert(a >= b);
            assert(a >= c);
            assert(a >= d);
            assert(a >= e);
            assert(a >= f);
            assert(a <= g);
            assert(a <= h);
            assert(a >= i);
            assert(b <= a);
            assert(b >= b);
            assert(b >= c);
            assert(b >= d);
            assert(b >= e);
            assert(b >= f);
            assert(b <= g);
            assert(b <= h);
            assert(b >= i);
            assert(c <= a);
            assert(c <= b);
            assert(c >= c);
            assert(c >= d);
            assert(c >= e);
            assert(c >= f);
            assert(c <= g);
            assert(c <= h);
            assert(c >= i);
            assert(d <= a);
            assert(d <= b);
            assert(d <= c);
            assert(d >= d);
            assert(d <= e);
            assert(d <= f);
            assert(d <= g);
            assert(d <= h);
            assert(d >= i);
            assert(e <= a);
            assert(e <= b);
            assert(e <= c);
            assert(e <= d);
            assert(e >= e);
            assert(e <= f);
            assert(e <= g);
            assert(e <= h);
            assert(e >= i);
            assert(f <= a);
            assert(f <= b);
            assert(f <= c);
            assert(f <= d);
            assert(f <= e);
            assert(f >= f);
            assert(f <= g);
            assert(f <= h);
            assert(f >= i);
            assert(g >= a);
            assert(g >= b);
            assert(g >= c);
            assert(g >= d);
            assert(g >= e);
            assert(g >= f);
            assert(g >= g);
            assert(g <= h);
            assert(g >= i);
            assert(h >= a);
            assert(h >= b);
            assert(h >= c);
            assert(h >= d);
            assert(h >= e);
            assert(h >= f);
            assert(h <= g);
            assert(h >= h);
            assert(h >= i);
            assert(i <= a);
            assert(i <= b);
            assert(i <= c);
            assert(i <= d);
            assert(i <= e);
            assert(i <= f);
            assert(i <= g);
            assert(i <= h);
            assert(i >= i);
            
            clearvars -except display;
        end
    end
end
