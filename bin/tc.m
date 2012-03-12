classdef tc
    methods (Static,Access=public)
        function [o] = any(i,classes)
            if exist('classes','var')
                o = tc.atomic(i,classes) || tc.object(i);
            else
                o = tc.atomic(i) || tc.object(i);
            end
        end
        
        function [o] = atomic(i,classes)
            if exist('classes','var')
                o = tc.value(i) || tc.labels(i) || tc.labels_idx(i,classes) || tc.cell(i) || tc.function_h(i);
            else
                o = tc.value(i) || tc.labels(i) || tc.cell(i) || tc.function_h(i);
            end
        end
        
        function [o] = value(i)
            o = tc.logical(i) || tc.number(i) || tc.string(i);
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
        
        function [o] = labels(i)
            o = tc.logical(i) || tc.natural(i) || (tc.cell(i) && all(cellfun(@tc.string,i)));
        end
        
        function [o] = labels_idx(i,classes)
            o = tc.natural(i) && tc.check(i > 0 & i <= length(classes));
        end
        
        function [o] = cell(i)
            o = iscell(i);
        end
        
        function [o] = function_h(i)
            o = isa(i,'function_handle');
        end
        
        function [o] = object(i)
            o = tc.samples(i) || tc.transform(i) || tc.classifier(i);
        end
        
        function [o] = samples_set(i)
            o = isa(i,'samples_set');
        end
        
        function [o] = gray_images_set(i)
            o = isa(i,'gray_images_set');
        end
        
        function [o] = transform(i)
            o = isa(i,'transform');
        end
        
        function [o] = classifier(i)
            o = isa(i,'classifier');
        end
                
        function [o] = empty(i)
            o = isempty(i);
        end
        
        function [o] = scalar(i)
            o = (length(size(i)) == 2) && (size(i,1) == 1) && (size(i,2) == 1);
        end
        
        function [o] = vector(i)
            o = (length(size(i)) == 2) && (((size(i,1) == 1) && (size(i,2) >= 1)) || ...
                                           ((size(i,1) >= 1) && (size(i,2) == 1)));
        end
        
        function [o] = matrix(i)
            o = (length(size(i)) == 2) && (size(i,1) >= 1) && (size(i,2) >= 1);
        end
        
        function [o] = tensor(i,d)
            if d == 0
                o = tc.scalar(i);
            elseif d == 1
                o = tc.vector(i);
            elseif d == 2
                o = tc.matrix(i);
            else
                o = (length(size(i)) <= d) && (all(size(i) >= 1));
            end
        end
        
        function [o] = match_dims(a,b,d1,d2)
            if exist('d1','var') && exist('d2','var')
                o = size(a,d1) == size(b,d2);
            elseif exist('d1','var')
                if tc.vector(a) && tc.vector(b)
                    o = size(a,d1) == size(b,d1);
                elseif tc.vector(a)
                    o = length(a) == size(b,d1);
                elseif tc.vector(b)
                    o = length(b) == size(a,d1);
                else
                    o = size(a,d1) == size(b,d1);
                end
            else
                if tc.vector(a) && tc.vector(b);
                    o = length(a) == length(b);
                else
                    o = size(a,1) == size(b,1);
                end
            end
        end
                
        function [o] = check(i)
            o = ~tc.empty(i) && all(i(:));
        end
    end
    
    methods (Static,Access=public)        
        function test(~)
            fprintf('Testing "tc".\n');
            
            % Tests for functions that check for specific types.
            
            fprintf('  Function "any".\n');
            
            assert(tc.any(true) == true);
            assert(tc.any(false) == true);
            assert(tc.any(10.3) == true);
            assert(tc.any(-4) == true);
            assert(tc.any(7) == true);
            assert(tc.any(0.3) == true);
            assert(tc.any('hello') == true);
            assert(tc.any({'0' '1' '2'}) == true);
            assert(tc.any([1 2 3],{'0' '1' '2'}) == true);
            assert(tc.any({0}) == true);
            assert(tc.any({1 2 3}) == true);
            assert(tc.any({'hello' 'world'}) == true);
            assert(tc.any(@()fprintf('hello')) == true);
            
            fprintf('  Function "atomic".\n');
            
            assert(tc.atomic(true) == true);
            assert(tc.atomic(false) == true);
            assert(tc.atomic(10.3) == true);
            assert(tc.atomic(-4) == true);
            assert(tc.atomic(7) == true);
            assert(tc.atomic(0.3) == true);
            assert(tc.atomic('hello') == true);
            assert(tc.atomic({'0' '1' '2'}) == true);
            assert(tc.atomic([1 2 3],{'0' '1' '2'}) == true);
            assert(tc.atomic({0}) == true);
            assert(tc.atomic({1 2 3}) == true);
            assert(tc.atomic({'hello' 'world'}) == true);
            assert(tc.atomic(@()fprintf('hello')) == true);
            
            fprintf('  Function "value".\n');
            
            assert(tc.value(true) == true);
            assert(tc.value(false) == true);
            assert(tc.value(10.3) == true);
            assert(tc.value(-4) == true);
            assert(tc.value(7) == true);
            assert(tc.value(0.3) == true);
            assert(tc.value('hello') == true);
            assert(tc.value({'0' '1' '2'}) == false);
            assert(tc.value({0}) == false);
            assert(tc.value({1 2 3}) == false);
            assert(tc.value({'hello' 'world'}) == false);
            assert(tc.value(@()fprintf('hello')) == false);
            
            fprintf('  Function "logical".\n');
            
            assert(tc.logical(true) == true);
            assert(tc.logical(false) == true);
            assert(tc.logical(10.3) == false);
            assert(tc.logical(-4) == false);
            assert(tc.logical(7) == false);
            assert(tc.logical(0.3) == false);
            assert(tc.logical('hello') == false);
            assert(tc.logical({'0' '1' '2'}) == false);
            assert(tc.logical({0}) == false);
            assert(tc.logical({1 2 3}) == false);
            assert(tc.logical({'hello' 'world'}) == false);
            assert(tc.logical(@()fprintf('hello')) == false);
            
            fprintf('  Function "number".\n');
            
            assert(tc.number(true) == false);
            assert(tc.number(false) == false);
            assert(tc.number(10.3) == true);
            assert(tc.number(-4) == true);
            assert(tc.number(7) == true);
            assert(tc.number(0.3) == true);
            assert(tc.number('hello') == false);
            assert(tc.number({'0' '1' '2'}) == false);
            assert(tc.number({0}) == false);
            assert(tc.number({1 2 3}) == false);
            assert(tc.number({'hello' 'world'}) == false);
            assert(tc.number(@()fprintf('hello')) == false);
            
            fprintf('  Function "integer".\n');
            
            assert(tc.integer(true) == false);
            assert(tc.integer(false) == false);
            assert(tc.integer(10.3) == false);
            assert(tc.integer(-4) == true);
            assert(tc.integer(7) == true);
            assert(tc.integer(0.3) == false);
            assert(tc.integer('hello') == false);
            assert(tc.integer({'0' '1' '2'}) == false);
            assert(tc.integer({0}) == false);
            assert(tc.integer({1 2 3}) == false);
            assert(tc.integer({'hello' 'world'}) == false);
            assert(tc.integer(@()fprintf('hello')) == false);
            
            fprintf('  Function "natural".\n');
            
            assert(tc.natural(true) == false);
            assert(tc.natural(false) == false);
            assert(tc.natural(10.3) == false);
            assert(tc.natural(-4) == false);
            assert(tc.natural(7) == true);
            assert(tc.natural(0.3) == false);
            assert(tc.natural('hello') == false);
            assert(tc.natural({'0' '1' '2'}) == false);
            assert(tc.natural({0}) == false);
            assert(tc.natural({1 2 3}) == false);
            assert(tc.natural({'hello' 'world'}) == false);
            assert(tc.natural(@()fprintf('hello')) == false);
            
            fprintf('  Function "unitreal".\n');
            
            assert(tc.unitreal(true) == false);
            assert(tc.unitreal(false) == false);
            assert(tc.unitreal(10.3) == false);
            assert(tc.unitreal(-4) == false);
            assert(tc.unitreal(7) == false);
            assert(tc.unitreal(0.3) == true);
            assert(tc.unitreal('hello') == false);
            assert(tc.unitreal({'0' '1' '2'}) == false);
            assert(tc.unitreal({0}) == false);
            assert(tc.unitreal({1 2 3}) == false);
            assert(tc.unitreal({'hello' 'world'}) == false);
            assert(tc.unitreal(@()fprintf('hello')) == false);
            
            fprintf('  Function "string".\n');
            
            assert(tc.string(true) == false);
            assert(tc.string(false) == false);
            assert(tc.string(10.3) == false);
            assert(tc.string(-4) == false);
            assert(tc.string(7) == false);
            assert(tc.string(0.3) == false);
            assert(tc.string('hello') == true);
            assert(tc.string({'0' '1' '2'}) == false);
            assert(tc.string({0}) == false);
            assert(tc.string({1 2 3}) == false);
            assert(tc.string({'hello' 'world'}) == false);
            assert(tc.string(@()fprintf('hello')) == false);
            
            fprintf('  Function "labels".\n');
            
            assert(tc.labels(true) == true);
            assert(tc.labels(false) == true);
            assert(tc.labels(10.3) == false);
            assert(tc.labels(-4) == false);
            assert(tc.labels(7) == true);
            assert(tc.labels(0.3) == false);
            assert(tc.labels('hello') == false);
            assert(tc.labels({'0' '1' '2'}) == true);
            assert(tc.labels({0}) == false);
            assert(tc.labels({1 2 3}) == false);
            assert(tc.labels({'hello' 'world'}) == true);
            assert(tc.labels(@()fprintf('hello')) == false);
            
            fprintf('  Function "labels_idx".\n');
            
            assert(tc.labels_idx(true,{'0' '1'}) == false);
            assert(tc.labels_idx(false,{'0' '1'}) == false);
            assert(tc.labels_idx(10.3,{'0' '1'}) == false);
            assert(tc.labels_idx(-4,{'0' '1'}) == false);
            assert(tc.labels_idx(7,{'0' '1'}) == false);
            assert(tc.labels_idx(0.3,{'0' '1'}) == false);
            assert(tc.labels_idx('hello',{'0' '1'}) == false);
            assert(tc.labels_idx({'0' '1' '2'},{'0' '1'}) == false);
            assert(tc.labels_idx([1 1 2],{'0' '1'}) == true);
            assert(tc.labels_idx({0},{'0' '1'}) == false);
            assert(tc.labels_idx({1 2 3},{'0' '1'}) == false);
            assert(tc.labels_idx({'hello' 'world'},{'0' '1'}) == false);
            assert(tc.labels_idx(@()fprintf('hello')) == false);
            
            fprintf('  Function "cell".\n');
            
            assert(tc.cell(true) == false);
            assert(tc.cell(false) == false);
            assert(tc.cell(10.3) == false);
            assert(tc.cell(-4) == false);
            assert(tc.cell(7) == false);
            assert(tc.cell(0.3) == false);
            assert(tc.cell('hello') == false);
            assert(tc.cell({'0' '1' '2'}) == true);
            assert(tc.cell({0}) == true);
            assert(tc.cell({1 2 3}) == true);
            assert(tc.cell({'hello' 'world'}) == true);
            assert(tc.cell(@()fprintf('hello')) == false);
            
            fprintf('  Function "function_h".\n');
            
            assert(tc.function_h(true) == false);
            assert(tc.function_h(false) == false);
            assert(tc.function_h(10.3) == false);
            assert(tc.function_h(-4) == false);
            assert(tc.function_h(7) == false);
            assert(tc.function_h(0.3) == false);
            assert(tc.function_h('hello') == false);
            assert(tc.function_h({'0' '1' '2'}) == false);
            assert(tc.function_h({0}) == false);
            assert(tc.function_h({1 2 3}) == false);
            assert(tc.function_h({'hello' 'world'}) == false);
            assert(tc.function_h(@()fprintf('hello')) == true);
             
            % Tests for functions that check for specific structure.
            
            fprintf('  Function "empty".\n');
            
            assert(tc.empty([]) == true);
            assert(tc.empty(zeros(1,0)) == true);
            assert(tc.empty(zeros(0,1)) == true);
            assert(tc.empty('') == true);
            assert(tc.empty({}) == true);
            assert(tc.empty(1) == false);
            assert(tc.empty([1 1 1]) == false);
            assert(tc.empty([1;1;1]) == false);
            assert(tc.empty(ones(4,3)) == false);
            assert(tc.empty(ones(4,3,4)) == false);
            assert(tc.empty({1}) == false);
            assert(tc.empty({1 1 1}) == false);
            assert(tc.empty({1;1;1}) == false);
            assert(tc.empty(num2cell(ones(4,3))) == false);
            assert(tc.empty(num2cell(ones(4,3,4))) == false);
            
            fprintf('  Function "scalar".\n');
            
            assert(tc.scalar([]) == false);
            assert(tc.scalar(zeros(1,0)) == false);
            assert(tc.scalar(zeros(0,1)) == false);
            assert(tc.scalar('') == false);
            assert(tc.scalar({}) == false);
            assert(tc.scalar(1) == true);
            assert(tc.scalar([1 1 1]) == false);
            assert(tc.scalar([1;1;1]) == false);
            assert(tc.scalar(ones(4,3)) == false);
            assert(tc.scalar(ones(4,3,4)) == false);
            assert(tc.scalar({1}) == true);
            assert(tc.scalar({1 1 1}) == false);
            assert(tc.scalar({1;1;1}) == false);
            assert(tc.scalar(num2cell(ones(4,3))) == false);
            assert(tc.scalar(num2cell(ones(4,3,4))) == false);
            
            fprintf('  Function "vector".\n');
            
            assert(tc.vector([]) == false);
            assert(tc.vector(zeros(1,0)) == false);
            assert(tc.vector(zeros(0,1)) == false);
            assert(tc.vector('') == false);
            assert(tc.vector({}) == false);
            assert(tc.vector(1) == true);
            assert(tc.vector([1 1 1]) == true);
            assert(tc.vector([1;1;1]) == true);
            assert(tc.vector(ones(4,3)) == false);
            assert(tc.vector(ones(4,3,4)) == false);
            assert(tc.vector({1}) == true);
            assert(tc.vector({1 1 1}) == true);
            assert(tc.vector({1;1;1}) == true);
            assert(tc.vector(num2cell(ones(4,3))) == false);
            assert(tc.vector(num2cell(ones(4,3,4))) == false);
            
            fprintf('  Function "matrix".\n');
            
            assert(tc.matrix([]) == false);
            assert(tc.matrix(zeros(1,0)) == false);
            assert(tc.matrix(zeros(0,1)) == false);
            assert(tc.matrix('') == false);
            assert(tc.matrix({}) == false);
            assert(tc.matrix(1) == true);
            assert(tc.matrix([1 1 1]) == true);
            assert(tc.matrix([1;1;1]) == true);
            assert(tc.matrix(ones(4,3)) == true);
            assert(tc.matrix(ones(4,3,4)) == false);
            assert(tc.matrix({1}) == true);
            assert(tc.matrix({1 1 1}) == true);
            assert(tc.matrix({1;1;1}) == true);
            assert(tc.matrix(num2cell(ones(4,3))) == true);
            assert(tc.matrix(num2cell(ones(4,3,4))) == false);
            
            fprintf('  Function "tensor".\n');
            
            assert(tc.tensor([],3) == false);
            assert(tc.tensor(zeros(1,0),3) == false);
            assert(tc.tensor(zeros(0,1),3) == false);
            assert(tc.tensor('',3) == false);
            assert(tc.tensor({},3) == false);
            assert(tc.tensor(1,3) == true);
            assert(tc.tensor([1 1 1],3) == true);
            assert(tc.tensor([1;1;1],3) == true);
            assert(tc.tensor(ones(4,3),3) == true);
            assert(tc.tensor(ones(4,3,4),3) == true);
            assert(tc.tensor({1},3) == true);
            assert(tc.tensor({1 1 1},3) == true);
            assert(tc.tensor({1;1;1},3) == true);
            assert(tc.tensor(num2cell(ones(4,3)),3) == true);
            assert(tc.tensor(num2cell(ones(4,3,4)),3) == true);
            
            fprintf('  Function "match_dims".\n');
            
            assert(tc.match_dims(1,1) == true);
            assert(tc.match_dims([1 2 3],[1 3 2]) == true);
            assert(tc.match_dims([1;2;3],[1 3 2]) == true);
            assert(tc.match_dims(zeros(4,3),zeros(4,3)) == true);
            assert(tc.match_dims(zeros(4,4),zeros(4,3)) == true);
            assert(tc.match_dims(1,[1 2]) == false);
            assert(tc.match_dims([1 2 3],[1 3 2 4]) == false);
            assert(tc.match_dims([1;2;3],[1 3 2 4]) == false);
            assert(tc.match_dims(zeros(4,2),zeros(3,2)) == false);
            assert(tc.match_dims(1,1,1) == true);
            assert(tc.match_dims(1,1,2) == true);
            assert(tc.match_dims(1,1,3) == true);
            assert(tc.match_dims([1 2 3],[1 3 2],1) == true);
            assert(tc.match_dims([1 2 3],[1 3 2],2) == true);
            assert(tc.match_dims([1 2 3],[1 3 2],3) == true);
            assert(tc.match_dims([1 2 3],zeros(3,4),1) == true);
            assert(tc.match_dims([1 2 3 4],zeros(3,4),2) == true);
            assert(tc.match_dims(zeros(3,4),[1 2 3],1) == true);
            assert(tc.match_dims(zeros(3,4),[1 2 3 4],2) == true);
            assert(tc.match_dims(zeros(3,3),zeros(3,4),1) == true);
            assert(tc.match_dims(zeros(3,4),zeros(2,4),2) == true);
            assert(tc.match_dims(1,[1 2],2) == false);
            assert(tc.match_dims([1 2 3],[1 3 2 4],2) == false);
            assert(tc.match_dims([1 2 3],zeros(3,4),2) == false);
            assert(tc.match_dims(zeros(3,4),zeros(4,3),1) == false);
            assert(tc.match_dims(zeros(3,4),zeros(4,3),2) == false);
            assert(tc.match_dims(1,1,1,1) == true);
            assert(tc.match_dims([1 2 3],[1 2 3],1,1) == true);
            assert(tc.match_dims([1;2;3],[1 2 3],1,2) == true);
            assert(tc.match_dims([1 2 3],[1 2 3],1,3) == true);
            assert(tc.match_dims([1;2;3],[1 2 3],1,2) == true);
            assert(tc.match_dims(zeros(3,4),zeros(5,3),1,2) == true);
            assert(tc.match_dims(zeros(3,4),zeros(3,5),1,1) == true);
            assert(tc.match_dims(1,[1 2],1,2) == false);
            assert(tc.match_dims([1;2;3],[1 2 3],1,1) == false);
            assert(tc.match_dims(zeros(4,3),zeros(4,3),1,2) == false);
            
            fprintf('  Function "check".\n');
            
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
            assert(tc.check([1 2 3]) == true);
            assert(tc.check([0 0 0]) == false);
            assert(tc.check(zeros(4,4,4) > -1 & zeros(4,4,4) < 1) == true);
        end
    end
end
