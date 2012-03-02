classdef tc
    methods (Static,Access=public)
        function [o] = any(~)
            o = true;
        end
        
        function [o] = atom(i)
            o = ~iscell(i);
        end
        
        function [o] = logical(i)
            o = islogical(i);
        end
        
        function [o] = number(i)
            o = isnumeric(i);
        end
        
        function [o] = integer(i)
            o = isnumeric(i) && tc.check(floor(i) == i);
        end
        
        function [o] = natural(i)
            o = isnumeric(i) && tc.check(floor(i) == i) && tc.check(i >= 0);
        end
        
        function [o] = unitreal(i)
            o = isfloat(i) && tc.check(i >= 0 & i <= 1);
        end
        
        function [o] = string(i)
            o = (isempty(i) || tc.vector(i)) && ischar(i);
        end
        
        function [o] = cell(i)
            o = iscell(i);
        end
        
        function [o] = empty(i)
            o = isempty(i);
        end
        
        function [o] = scalar(i)
            o = (length(size(i)) == 2) && (size(i,1) == 1) && (size(i,2) == 1);
        end
        
        function [o] = vector(i)
            o = tc.vector_row(i) || tc.vector_col(i);
        end
        
        function [o] = vector_row(i)
            o = (length(size(i)) == 2) && (size(i,1) == 1) && (size(i,2) >= 1);
        end
        
        function [o] = vector_col(i)
            o = (length(size(i)) == 2) && (size(i,1) >= 1) && (size(i,2) == 1);
        end
        
        function [o] = matrix(i)
            o = (length(size(i)) == 2) && (size(i,1) >= 1) && (size(i,2) >= 1);
        end
                
        function [o] = match_cols(a,b)
            o = (length(size(a)) == 2) && (length(size(b)) == 2) && (size(a,2) == size(b,2));
        end
        
        function [o] = match_rows(a,b)
            o = (length(size(a)) == 2) && (length(size(b)) == 2) && (size(a,1) == size(b,1));
        end
        
        function [o] = match_size(a,b)
            o = (length(size(a)) == 2) && (length(size(b)) == 2) && ...
                (size(a,1) == size(b,1)) && (size(a,2) == size(b,2));
        end
        
        function [o] = check(i)
            o = ~tc.empty(i) && all(all(i));
        end
        
        function [o] = labels_idx(i,classes,samples)
            o = tc.vector_col(i) && tc.match_rows(samples,i) && ...
                tc.natural(i) && tc.check(i > 0 & i <= length(classes));
        end
        
        function [o] = labels(i,samples)
            o = tc.vector_col(i) && tc.match_rows(samples,i) && ...
                (tc.logical(i) || tc.natural(i) || (tc.cell(i) && all(cellfun(@tc.string,i))));
        end
    end
    
    methods (Static,Access=public)        
        function unittest            
            % Tests for functions that check for specific types.
            
            fprintf('Testing function "any".\n');
            
            assert(tc.any(true) == true);
            assert(tc.any(false) == true);
            assert(tc.any(10.3) == true);
            assert(tc.any(-3) == true);
            assert(tc.any(4) == true);
            assert(tc.any('hello') == true);
            assert(tc.any(struct('hello',10,'world',20)) == true);
            assert(tc.any([true false]) == true);
            assert(tc.any(true(10,4)) == true);
            assert(tc.any([3.1 4 5.5]) == true);
            assert(tc.any(3.4*ones(4,3)) == true);
            assert(tc.any(['hello';'world']) == true);
            assert(tc.any([struct('hello',10,'world',20); struct('hello',100,'world',200)]) == true);
            assert(tc.any({true}) == true);
            assert(tc.any({10.3}) == true);
            assert(tc.any({-7}) == true);
            assert(tc.any({5}) == true);
            assert(tc.any({0.2}) == true);
            assert(tc.any({'hello'}) == true);
            assert(tc.any({true true; false true}) == true);
            assert(tc.any({'hello' 'world'}) == true);
            assert(tc.any({struct('hello',10,'world',20); struct('hello',100,'world',200)}) == true);
            assert(tc.any({[10 20] [10 20; 30 40]}) == true);
            
            fprintf('Testing function "atom".\n');
            
            assert(tc.atom(true) == true);
            assert(tc.atom(false) == true);
            assert(tc.atom(10.3) == true);
            assert(tc.atom(-3) == true);
            assert(tc.atom(4) == true);
            assert(tc.atom(0.2) == true);
            assert(tc.atom('hello') == true);
            assert(tc.atom(struct('hello',10,'world',20)) == true);
            assert(tc.atom([true false true]) == true);
            assert(tc.atom([2 3 1.3]) == true);
            assert(tc.atom(zeros(10,20)) == true);
            assert(tc.atom(['hello';'world']) == true);
            assert(tc.atom([struct('hello',10,'world',20); struct('hello',100,'world',200)]) == true);
            assert(tc.atom({true}) == false);
            assert(tc.atom({10.3}) == false);
            assert(tc.atom({-7}) == false);
            assert(tc.atom({5}) == false);
            assert(tc.atom({0.2}) == false);
            assert(tc.atom({'hello'}) == false);
            assert(tc.atom({true true; false true}) == false);
            assert(tc.atom({'hello' 'world'}) == false);
            assert(tc.atom({struct('hello',10,'world',20); struct('hello',100,'world',200)}) == false);
            assert(tc.atom({[10 20] [10 20; 30 40]}) == false);
            
            fprintf('Testing function "logical".\n');
            
            assert(tc.logical(true) == true);
            assert(tc.logical(false) == true);
            assert(tc.logical([true true false]) == true);
            assert(tc.logical([true false true]') == true);
            assert(tc.logical(true(10,10)) == true);
            assert(tc.logical(false(5,7)) == true);
            assert(tc.logical(1) == false);
            assert(tc.logical(0) == false);
            assert(tc.logical(3.43) == false);
            assert(tc.logical(-3) == false);
            assert(tc.logical(3) == false);
            assert(tc.logical(0.3) == false);
            assert(tc.logical('hello') == false);
            assert(tc.logical(struct('hello',10,'world',20)) == false);
            assert(tc.logical({true}) == false);
            assert(tc.logical({false}) == false);
            assert(tc.logical([1,2,3]) == false);
            assert(tc.logical(ones(10,10)) == false);
            
            fprintf('Testing function "number".\n');
            
            assert(tc.number(10) == true);
            assert(tc.number(10.34) == true);
            assert(tc.number(-3.23) == true);
            assert(tc.number(3e+7) == true);
            assert(tc.number([1 2 3]) == true);
            assert(tc.number([1 3 0.2 5]') == true);
            assert(tc.number(3.4*ones(10,4)) == true);
            assert(tc.number(true) == false);
            assert(tc.number(false) == false);
            assert(tc.number('hello') == false);
            assert(tc.number(struct('hello',10,'world',20)) == false);
            assert(tc.number({3.2}) == false);
            assert(tc.number({7 2 -3 1.2}) == false);
            
            fprintf('Testing function "integer".\n');
            
            assert(tc.integer(10) == true);
            assert(tc.integer(-3) == true);
            assert(tc.integer(0) == true);
            assert(tc.integer([1 2 3]) == true);
            assert(tc.integer([3 2 -1]') == true);
            assert(tc.integer(3*ones(10,20)) == true);
            assert(tc.integer(3.2) == false);
            assert(tc.integer(-0.3) == false);
            assert(tc.integer(true) == false);
            assert(tc.integer(false) == false);
            assert(tc.integer('hello') == false);
            assert(tc.integer(struct('hello',10,'world',20)) == false);
            assert(tc.integer({1}) == false);
            assert(tc.integer({1 -2 -3}) == false);
            
            fprintf('Testing function "natural".\n');
            
            assert(tc.natural(10) == true);
            assert(tc.natural(0) == true);
            assert(tc.natural([1 2 3]) == true);
            assert(tc.natural([3 2 1 4']) == true);
            assert(tc.natural(3*ones(4,10)) == true);
            assert(tc.natural(3.2) == false);
            assert(tc.natural(-0.3) == false);
            assert(tc.natural(-3) == false);
            assert(tc.natural(true) == false);
            assert(tc.natural(false) == false);
            assert(tc.natural('hello') == false);
            assert(tc.natural(struct('hello',10,'world',20)) == false);
            assert(tc.natural({1}) == false);
            assert(tc.natural({1 3 2}) == false);
            
            fprintf('Testing function "unitreal".\n');
            
            assert(tc.unitreal(0.1) == true);
            assert(tc.unitreal(0) == true);
            assert(tc.unitreal(1) == true);
            assert(tc.unitreal([0 1 0.3 0.2]) == true);
            assert(tc.unitreal([0.3 0.2 0]') == true);
            assert(tc.unitreal(rand(10,5)) == true);
            assert(tc.unitreal(2) == false);
            assert(tc.unitreal(-0.3) == false);
            assert(tc.unitreal(1.0001) == false);
            assert(tc.unitreal(true) == false);
            assert(tc.unitreal(false) == false);
            assert(tc.unitreal('hello') == false);
            assert(tc.unitreal(struct('hello',10,'world',20)) == false);
            assert(tc.unitreal({0.3}) == false);
            assert(tc.unitreal({0.2 0.1}) == false);
            
            fprintf('Testing function "string".\n');
            
            assert(tc.string('hello') == true);
            assert(tc.string(transpose('hello')) == true);
            assert(tc.string('') == true);
            assert(tc.string(['hello';'world']) == false);
            assert(tc.string(true) == false);
            assert(tc.string(false) == false);
            assert(tc.string(3.7) == false);
            assert(tc.string(10) == false);
            assert(tc.string(-3.2) == false);
            assert(tc.string(0.3) == false);
            assert(tc.string(struct('hello',10,'world',20)) == false);
            assert(tc.string({'hello'}) == false);
            assert(tc.string({'hello' 'world'}) == false);
            
            fprintf('Testing function "cell".\n');
            
            assert(tc.cell({true}) == true);
            assert(tc.cell({10.3}) == true);
            assert(tc.cell({-7}) == true);
            assert(tc.cell({5}) == true);
            assert(tc.cell({0.2}) == true);
            assert(tc.cell({'hello'}) == true);
            assert(tc.cell({true true; false true}) == true);
            assert(tc.cell({'hello' 'world'}) == true);
            assert(tc.cell({struct('hello',10,'world',20); struct('hello',100,'world',200)}) == true);
            assert(tc.cell({[10 20] [10 20; 30 40]}) == true);
            assert(tc.cell(true) == false);
            assert(tc.cell(false) == false);
            assert(tc.cell(10.3) == false);
            assert(tc.cell(-3) == false);
            assert(tc.cell(4) == false);
            assert(tc.cell(0.2) == false);
            assert(tc.cell('hello') == false);
            assert(tc.cell(struct('hello',10,'world',20)) == false);
            assert(tc.cell([true false true]) == false);
            assert(tc.cell([2 3 1.3]) == false);
            assert(tc.cell(zeros(10,20)) == false);
            
            % Tests for functions that check for specific structure.
            
            fprintf('Testing function "empty".\n');
            
            assert(tc.empty([]) == true);
            assert(tc.empty(zeros(1,0)) == true);
            assert(tc.empty(zeros(0,1)) == true);
            assert(tc.empty('') == true);
            assert(tc.empty({}) == true);
            assert(tc.empty(true) == false);
            assert(tc.empty(false) == false);
            assert(tc.empty(10.3) == false);
            assert(tc.empty(-3) == false);
            assert(tc.empty(4) == false);
            assert(tc.empty(0.3) == false);
            assert(tc.empty('hello') == false);
            assert(tc.empty(struct('hello',10,'world',20)) == false);
            assert(tc.empty(true(20,2)) == false);
            assert(tc.empty([1 3 0.4]) == false);
            assert(tc.empty(['hello';'world']) == false);
            assert(tc.empty([struct('hello',10,'world',20) struct('hello',10,'world',20)]) == false);
            assert(tc.empty({true false}) == false);
            assert(tc.empty({10 20; 30 40}) == false);
            assert(tc.empty({'hello' 'world'; 'how' 'are you today?'}) == false);
            
            fprintf('Testing function "scalar".\n');
            
            assert(tc.scalar(true) == true);
            assert(tc.scalar(10.32) == true);
            assert(tc.scalar(-3) == true);
            assert(tc.scalar(20) == true);
            assert(tc.scalar(0.3) == true);
            assert(tc.scalar(ones(1,1,1,1)) == true);
            assert(tc.scalar(true(1,1,1,1)) == true);
            assert(tc.scalar(false(1,1,1,1)) == true);
            assert(tc.scalar(struct('hello',10,'world',20)) == true);
            assert(tc.scalar([1 2]) == false);
            assert(tc.scalar([1 2 3]') == false);
            assert(tc.scalar(3.2*ones(10,20)) == false);
            assert(tc.scalar('hello') == false);
            assert(tc.scalar([struct('hello',10,'world',20) struct('hello',100,'world',200)]) == false);
            
            fprintf('Testing function "vector".\n');
            
            assert(tc.vector(true) == true);
            assert(tc.vector(10.32) == true);
            assert(tc.vector(-3) == true);
            assert(tc.vector(20) == true);
            assert(tc.vector(0.3) == true);
            assert(tc.vector([true false false]) == true);
            assert(tc.vector([true false false]') == true);
            assert(tc.vector([1.3 2.6 3]) == true);
            assert(tc.vector([1.3 2.6 3]') == true);
            assert(tc.vector([-54 -3 21]) == true);
            assert(tc.vector([-54 -3 21]') == true);
            assert(tc.vector([1 3 2 7]) == true);
            assert(tc.vector([1 3 2 7]') == true);
            assert(tc.vector([struct('hello',10,'world',20) struct('hello',100,'world',200)]) == true);
            assert(tc.vector('hello') == true);
            assert(tc.vector(3.2*ones(100,1)) == true);
            assert(tc.vector(3.4*ones(1,100)) == true);
            assert(tc.vector(ones(100,10)) == false);
            assert(tc.vector(ones(0,0)) == false);
            
            fprintf('Testing function "vector_row".\n');
            
            assert(tc.vector_row(true) == true);
            assert(tc.vector_row(10.32) == true);
            assert(tc.vector_row(-3) == true);
            assert(tc.vector_row(20) == true);
            assert(tc.vector_row(0.3) == true);
            assert(tc.vector_row([true false false]) == true);
            assert(tc.vector_row([1.3 2.6 3]) == true);
            assert(tc.vector_row([-54 -3 21]) == true);
            assert(tc.vector_row([1 3 2 7]) == true);
            assert(tc.vector_row('hello') == true);
            assert(tc.vector_row(3.4*ones(1,100)) == true);
            assert(tc.vector_row([struct('hello',10,'world',20) struct('hello',100,'world',200)]) == true);
            assert(tc.vector_row([true false false]') == false);
            assert(tc.vector_row([1.3 2.6 3]') == false);
            assert(tc.vector_row([-54 -3 21]') == false);
            assert(tc.vector_row([1 3 2 7]') == false);
            assert(tc.vector_row([struct('hello',10,'world',20) struct('hello',100,'world',200)]') == false);
            assert(tc.vector_row(3.2*ones(100,1)) == false);
            assert(tc.vector_row(ones(100,10)) == false);
            assert(tc.vector_row(ones(0,0)) == false);
            
            fprintf('Testing function "vector_col".\n');
            
            assert(tc.vector_col(true) == true);
            assert(tc.vector_col(10.32) == true);
            assert(tc.vector_col(-3) == true);
            assert(tc.vector_col(20) == true);
            assert(tc.vector_col(0.3) == true);
            assert(tc.vector_col([true;false;false]) == true);
            assert(tc.vector_col([1.3;2;6;3]) == true);
            assert(tc.vector_col([-54;-3;21]) == true);
            assert(tc.vector_col([1;3;2;7]) == true);
            assert(tc.vector_col(3.4*ones(1,100)) == false);
            assert(tc.vector_col([struct('hello',10,'world',20);struct('hello',100,'world',200)]) == true);
            assert(tc.vector_col([true false false]) == false);
            assert(tc.vector_col([1.3 2.6 3]) == false);
            assert(tc.vector_col([-54 -3 21]) == false);
            assert(tc.vector_col([1 3 2 7]) == false);
            assert(tc.vector_col('hello') == false);
            assert(tc.vector_col([struct('hello',10,'world',20) struct('hello',100,'world',200)]) == false);
            assert(tc.vector_col(3.2*ones(100,1)) == true);
            assert(tc.vector_col(ones(100,10)) == false);
            assert(tc.vector_col(ones(0,0)) == false);
            
            fprintf('Testing function "matrix".\n');
            
            assert(tc.matrix(true) == true);
            assert(tc.matrix(10.32) == true);
            assert(tc.matrix(-3) == true);
            assert(tc.matrix(20) == true);
            assert(tc.matrix(0.3) == true);
            assert(tc.matrix([true false false]) == true);
            assert(tc.matrix([true false false]') == true);
            assert(tc.matrix([1.3 2.6 3]) == true);
            assert(tc.matrix([1.3 2.6 3]') == true);
            assert(tc.matrix([-54 -3 21]) == true);
            assert(tc.matrix([-54 -3 21]') == true);
            assert(tc.matrix([1 3 2 7]) == true);
            assert(tc.matrix([1 3 2 7]') == true);
            assert(tc.matrix([struct('hello',10,'world',20) struct('hello',100,'world',200)]) == true);
            assert(tc.matrix('hello') == true);
            assert(tc.matrix(3.2*ones(100,1)) == true);
            assert(tc.matrix(3.4*ones(1,100)) == true);
            assert(tc.matrix([true true; false true]) == true);
            assert(tc.matrix([1.3 2.2; 1.4 7.3]) == true);
            assert(tc.matrix([-3 -2 -1; 7 3 -4; -6 1 9.3]) == true);
            assert(tc.matrix([1 2 3; 3 2 1; 1 3 2]) == true);
            assert(tc.matrix([0.1 0.1; 0.1 0.3; 0.03 0.99]) == true);
            assert(tc.matrix(['hello';'world']) == true);
            assert(tc.matrix([struct('hello',10,'world',20) struct('hello',100,'world',200);
                struct('hello',1,'world',2) struct('hello',0.1,'world',0.2)]) == true);
            
            % Test functions that check particular object properties.
            
            fprintf('Testing function "match_rows".\n');
            
            assert(tc.match_rows([1 2; 2 3; 1 4],[1;3;2]) == true);
            assert(tc.match_rows(rand(5,3),rand(5,1)) == true);
            assert(tc.match_rows(rand(3,2),rand(3,3)) == true);
            assert(tc.match_rows(rand(4,2),rand(1,1)) == false);
            assert(tc.match_rows(rand(4,2),rand(3,1)) == false);
            
            fprintf('Testing function "match_cols".\n');
            
            assert(tc.match_cols([1 2; 2 3; 1 4],[1 2; 3 1]) == true);
            assert(tc.match_cols(rand(5,3),rand(4,3)) == true);
            assert(tc.match_cols(rand(3,2),rand(1,2)) == true);
            assert(tc.match_cols(rand(4,2),rand(1,3)) == false);
            assert(tc.match_cols(rand(4,2),rand(4,3)) == false);
            
            fprintf('Testing function "match_size".\n');
            
            assert(tc.match_size([1 2; 3 2],[1 4; 2 8]) == true);
            assert(tc.match_size(rand(5,5),rand(5,5)) == true);
            assert(tc.match_size(rand(3,2),rand(3,2)) == true);
            assert(tc.match_size(rand(2,3),rand(3,2)) == false);
            assert(tc.match_size(rand(5,8),rand(2,8)) == false);
            
            fprintf('Testing function "check".\n');
            
            assert(tc.check(true) == true);
            assert(tc.check(false) == false);
            assert(tc.check(7.3) == true);
            assert(tc.check(-5) == true);
            assert(tc.check(4) == true);
            assert(tc.check(0) == false);
            assert(tc.check(10) == true);
            assert(tc.check('hello') == true);
            assert(tc.check(true(4,2)) == true);
            assert(tc.check(false(5,3)) == false);
            assert(tc.check(7.3*ones(4,9)) == true);
            assert(tc.check(-5*ones(3,2)) == true);
            assert(tc.check(4*ones(4,4)) == true);
            assert(tc.check(['hello';'world']) == true);
            assert(tc.check(find([1 2 3])) == true);
            assert(tc.check(tc.match_size(rand(8,3),rand(8,3))) == true);
            assert(tc.check(find([0 0 0])) == false);
            
            % Testing specialized functions.
            
            fprintf('Testing function "labels_idx".\n');
            
            assert(tc.labels_idx(1,{'1','2'},rand(1,4)) == true);
            assert(tc.labels_idx(2,{'1','2'},rand(1,4)) == true);
            assert(tc.labels_idx([1;1;2],{'1','2'},rand(3,4)) == true);
            assert(tc.labels_idx([1;2;1;2;2;3],{'1','2','3'},rand(6,4)) == true);
            assert(tc.labels_idx(3,{'1','2'},rand(1,4)) == false);
            assert(tc.labels_idx([1 2],{'1','2'},rand(2,4)) == false);
            assert(tc.labels_idx([9.3 0.2],{'1','2'},rand(2,4)) == false);
            assert(tc.labels_idx('hello',{'1','2'},rand(1,5)) == false);
            assert(tc.labels_idx({1;2;3},{'1','2'},rand(3,4)) == false);
            assert(tc.labels_idx([1;1;2],{'1','2'},rand(2,4)) == false);
            assert(tc.labels_idx({'hello';'world'},{'hello','world'},rand(2,4)) == false);
            assert(tc.labels_idx(struct('hello',10,'world',20),{'1','2'},rand(1,4)) == false);
            assert(tc.labels_idx(ones(5,5),{'1','2'},rand(5,5)) == false);
            
            fprintf('Testing function "labels".\n');
            
            assert(tc.labels(true,rand(1,4)) == true);
            assert(tc.labels(false,rand(1,4)) == true);
            assert(tc.labels([true;true;false],rand(3,5)) == true);
            assert(tc.labels(0,rand(1,4)) == true);
            assert(tc.labels(7,rand(1,5)) == true);
            assert(tc.labels([0;2;7;5],rand(4,5)) == true);
            assert(tc.labels({'hello';'world'},rand(2,5)) == true);
            assert(tc.labels([1 2],rand(2,3)) == false);
            assert(tc.labels({'hello' 'world'},rand(2,4)) == false);
            assert(tc.labels(true(4,3),rand(4,4)) == false);
            assert(tc.labels([1;3;2],rand(2,4)) == false);
            assert(tc.labels(0.3,rand(1,4)) == false);
            assert(tc.labels(-5,rand(1,4)) == false);
            assert(tc.labels([2;3;-3],rand(3,4)) == false);
            assert(tc.labels([2;3;0.3;4],rand(4,4)) == false);
            assert(tc.labels('hello',rand(1,4)) == false);
            assert(tc.labels(struct('hello',10,'world',20),rand(1,4)) == false);
            assert(tc.labels({1;2;3},rand(3,2)) == false);
        end
    end
end
