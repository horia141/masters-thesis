classdef check
    methods (Static,Access=public)
        function [o] = any(obj)
            o = check.atomic(obj) || check.object(obj);
        end
        
        function [o] = atomic(obj)
            o = check.value(obj) || check.cell(obj);
        end
        
        function [o] = value(obj)
            o = check.logical(obj) || check.number(obj) || check.string(obj) || check.function_h(obj) || check.struct(obj);
        end
        
        function [o] = logical(obj)
            o = islogical(obj);
        end
        
        function [o] = number(obj)
            o = isnumeric(obj);
        end
        
        function [o] = integer(obj)
            o = isnumeric(obj) && check.checkv(floor(obj) == obj);
        end
        
        function [o] = natural(obj)
            o = isnumeric(obj) && check.checkv(floor(obj) == obj) && check.checkv(obj >= 0);
        end
        
        function [o] = unitreal(obj)
            o = isfloat(obj) && check.checkv(obj >= 0 & obj <= 1);
        end
        
        function [o] = string(obj)
            o = ischar(obj);
        end
        
        function [o] = function_h(obj)
            o = isa(obj,'function_handle');
        end
        
        function [o] = struct(obj,varargin)
            o = isstruct(obj) && (check.empty(varargin) || all(isfield(obj,varargin)));
        end
        
        function [o] = cell(obj)
            o = iscell(obj);
        end
        
        function [o] = object(obj)
            o = check.dataset(obj) || ...
                check.classifier(obj) || check.classifier_info(obj) || ...
                check.regressor(obj) || check.regressor_info(obj) || ...
                check.transform(obj) || ...
                check.logging_logger(obj) || check.logging_handler(obj) || check.logging_level(obj);
        end
        
        function [o] = dataset(obj)
            o = check.dataset_record(obj) || check.dataset_image(obj);
        end
        
        function [o] = dataset_record(obj)
            o = check.matrix(obj) && check.number(obj);
        end
        
        function [o] = dataset_image(obj)
            o = check.tensor(obj,4) && check.number(obj);
        end
        
        function [o] = classifier(obj)
            o = isa(obj,'classifier');
        end
        
        function [o] = classifier_info(obj)
            o = isa(obj,'classifier_info');
        end
        
        function [o] = regressor(obj)
            o = isa(obj,'regressor');
        end
        
        function [o] = regressor_info(obj)
            o = isa(obj,'regressor_info');
        end
        
        function [o] = transform(obj)
            o = isa(obj,'transform');
        end
        
        function [o] = logging_logger(obj)
            o = isa(obj,'logging.logger');
        end
        
        function [o] = logging_handler(obj)
            o = isa(obj,'logging.handler');
        end
        
        function [o] = logging_level(obj)
            o = isa(obj,'logging.level');
        end
        
        function [o] = empty(obj)
            o = isempty(obj);
        end
        
        function [o] = scalar(obj)
            if check.string(obj)
                o = (length(size(obj)) == 2) && ((size(obj,1) == 1) && (size(obj,2) >= 1));
            else
                o = (length(size(obj)) == 2) && (size(obj,1) == 1) && (size(obj,2) == 1);
            end
        end
        
        function [o] = vector(obj)
            if check.string(obj)
                o = false;
            else
                o = (length(size(obj)) == 2) && (((size(obj,1) == 1) && (size(obj,2) >= 1)) || ...
                                                 ((size(obj,1) >= 1) && (size(obj,2) == 1)));
            end
        end
        
        function [o] = matrix(obj)
            if check.string(obj)
                o = false;
            else
                o = (length(size(obj)) == 2) && (size(obj,1) >= 1) && (size(obj,2) >= 1);
            end
        end
        
        function [o] = tensor(obj,d)
            if ~exist('d','var')
                o = ~check.empty(obj);
            else
                if check.string(obj)
                    if d == 1
                        o = true;
                    else
                        o = false;
                    end
                else
                    if d == 0
                        o = check.scalar(obj);
                    elseif d == 1
                        o = check.vector(obj);
                    elseif d == 2
                        o = check.matrix(obj);
                    else
                        o = (length(size(obj)) <= d) && (all(size(obj) >= 1));
                    end
                end
            end
        end
        
        function [o] = match_dims(obj1,obj2,d1,d2)
            if exist('d1','var') && exist('d2','var')
                o = size(obj1,d1) == size(obj2,d2);
            elseif exist('d1','var')
                if check.vector(obj1) && check.vector(obj2)
                    o = size(obj1,d1) == size(obj2,d1);
                elseif check.vector(obj1)
                    o = length(obj1) == size(obj2,d1);
                elseif check.vector(obj2)
                    o = length(obj2) == size(obj1,d1);
                else
                    o = size(obj1,d1) == size(obj2,d1);
                end
            else
                if check.vector(obj1) && check.vector(obj2);
                    o = length(obj1) == length(obj2);
                else
                    o = size(obj1,1) == size(obj2,1);
                end
            end
        end
                
        function [o] = checkv(obj,empty_good)
            if exist('empty_good','var')
                empty_good_t = empty_good;
            else
                empty_good_t = false;
            end
            
            if ~check.empty(obj)
                o_t = all(obj(:));
            else
                o_t = false;
            end
            
            if empty_good_t
                o = check.empty(obj) || o_t;
            else
                o = ~check.empty(obj) && o_t;
            end
        end
        
        function [o] = checkf(check_fn,obj,empty_good)
            if exist('empty_good','var')
                empty_good_t = empty_good;
            else
                empty_good_t = false;
            end
            
            if check.cell(obj)
                o_t = all(cellfun(check_fn,obj(:)));
            else
                o_t = all(arrayfun(check_fn,obj(:)));
            end
            
            if empty_good_t
                o = check.empty(obj) || o_t;
            else
                o = ~check.empty(obj) && o_t;
            end
        end
        
        function [o] = same(obj1,obj2,approx_epsilon)
            if ~exist('approx_epsilon','var')
                approx_epsilon = 1e-6;
            end
            
            if check.empty(obj1) && check.empty(obj2)
                o = true;
            else
                assert((check.tensor(obj1) && check.logical(obj1)) || ...
                       (check.tensor(obj1) && check.number(obj1)) || ...
                       (check.scalar(obj1) && check.string(obj1)) || ...
                       (check.scalar(obj1) && check.function_h(obj1)) || ...
                       (check.vector(obj1) && check.struct(obj1)) || ...
                       (check.vector(obj1) && check.cell(obj1)) || ...
                       (check.vector(obj1) && check.object(obj1)));
                assert((check.tensor(obj2) && check.logical(obj2)) || ...
                       (check.tensor(obj2) && check.number(obj2)) || ...
                       (check.scalar(obj2) && check.string(obj2)) || ...
                       (check.scalar(obj2) && check.function_h(obj2)) || ...
                       (check.vector(obj2) && check.struct(obj2)) || ...
                       (check.vector(obj2) && check.cell(obj2)) || ...
                       (check.vector(obj2) && check.object(obj2)));

                o = true;
                o = o && check.checkv(size(obj1) == size(obj2));
                o = o && ((check.logical(obj1) && check.logical(obj2)) || ...
                          (check.number(obj1) && check.number(obj2)) || ...
                          (check.string(obj1) && check.string(obj2)) || ...
                          (check.function_h(obj1) && check.function_h(obj2)) || ...
                          (check.struct(obj1) && check.struct(obj2)) || ...
                          (check.cell(obj1) && check.cell(obj2)) || ...
                          (check.object(obj1) && check.object(obj2)));

                if o
                    if check.logical(obj1)
                        o = check.checkv(obj1 == obj2);
                    elseif check.number(obj1)
                        o = check.checkv(abs(obj1 - obj2) < approx_epsilon);
                    elseif check.string(obj1)
                        o = strcmp(obj1,obj2);
                    elseif check.function_h(obj1)
                        o = strcmp(func2str(obj1),func2str(obj2));
                    elseif check.struct(obj1)
                        obj1p = orderfields(obj1);
                        obj2p = orderfields(obj2);

                        f1 = fieldnames(obj1p);
                        f2 = fieldnames(obj2p);

                        o = check.same(f1,f2);

                        if o
                            o = check.checkv(arrayfun(@(s1,s2)check.checkv(cellfun(@check.same,struct2cell(s1),struct2cell(s2))),obj1p,obj2p));
                        end
                    elseif check.cell(obj1)
                        o = check.checkv(cellfun(@check.same,obj1,obj2));
                    elseif check.object(obj1)
                        o = check.checkv(arrayfun(@(ii)obj1(ii) == obj2(ii),1:length(obj1)));
                    else
                        assert(false);
                    end
                end
            end
        end
        
        function [o] = one_of(obj,varargin)
            o = ~check.empty(find(cellfun(@(c)check.same(obj,c),varargin),1));
        end
    end
    
    methods (Static,Access=public)        
        function test(~)
            fprintf('Testing "check".\n');
            
            % Tests for functions that check for specific types.
            
            fprintf('  Function "any".\n');
            
            assert(check.any(true) == true);
            assert(check.any(false) == true);
            assert(check.any(10.3) == true);
            assert(check.any(-4) == true);
            assert(check.any(7) == true);
            assert(check.any(0.3) == true);
            assert(check.any('hello') == true);
            assert(check.any(@()fprintf('hello')) == true);
            assert(check.any(struct('hello',10,'world',20)) == true);
            assert(check.any({'0' '1' '2'}) == true);
            
            clearvars -except test_figure;
            
            fprintf('  Function "atomic".\n');
            
            assert(check.atomic(true) == true);
            assert(check.atomic(false) == true);
            assert(check.atomic(10.3) == true);
            assert(check.atomic(-4) == true);
            assert(check.atomic(7) == true);
            assert(check.atomic(0.3) == true);
            assert(check.atomic('hello') == true);
            assert(check.atomic(@()fprintf('hello')) == true);
            assert(check.atomic(struct('hello',10,'world',20)) == true);
            assert(check.atomic({'0' '1' '2'}) == true);
            
            clearvars -except test_figure;
            
            fprintf('  Function "value".\n');
            
            assert(check.value(true) == true);
            assert(check.value(false) == true);
            assert(check.value(10.3) == true);
            assert(check.value(-4) == true);
            assert(check.value(7) == true);
            assert(check.value(0.3) == true);
            assert(check.value('hello') == true);
            assert(check.value(@()fprintf('hello')) == true);
            assert(check.value(struct('hello',10,'world',20)) == true);
            assert(check.value({'0' '1' '2'}) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "logical".\n');
            
            assert(check.logical(true) == true);
            assert(check.logical(false) == true);
            assert(check.logical(10.3) == false);
            assert(check.logical(-4) == false);
            assert(check.logical(7) == false);
            assert(check.logical(0.3) == false);
            assert(check.logical('hello') == false);
            assert(check.logical(@()fprintf('hello')) == false);
            assert(check.logical(struct('hello',10,'world',20)) == false);
            assert(check.logical({'0' '1' '2'}) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "number".\n');
            
            assert(check.number(true) == false);
            assert(check.number(false) == false);
            assert(check.number(10.3) == true);
            assert(check.number(-4) == true);
            assert(check.number(7) == true);
            assert(check.number(0.3) == true);
            assert(check.number('hello') == false);
            assert(check.number(@()fprintf('hello')) == false);
            assert(check.number(struct('hello',10,'world',20)) == false);
            assert(check.number({'0' '1' '2'}) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "integer".\n');
            
            assert(check.integer(true) == false);
            assert(check.integer(false) == false);
            assert(check.integer(10.3) == false);
            assert(check.integer(-4) == true);
            assert(check.integer(7) == true);
            assert(check.integer(0.3) == false);
            assert(check.integer('hello') == false);
            assert(check.integer(@()fprintf('hello')) == false);
            assert(check.integer(struct('hello',10,'world',20)) == false);
            assert(check.integer({'0' '1' '2'}) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "natural".\n');
            
            assert(check.natural(true) == false);
            assert(check.natural(false) == false);
            assert(check.natural(10.3) == false);
            assert(check.natural(-4) == false);
            assert(check.natural(7) == true);
            assert(check.natural(0.3) == false);
            assert(check.natural('hello') == false);
            assert(check.natural(@()fprintf('hello')) == false);
            assert(check.natural(struct('hello',10,'world',20)) == false);
            assert(check.natural({'0' '1' '2'}) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "unitreal".\n');
            
            assert(check.unitreal(true) == false);
            assert(check.unitreal(false) == false);
            assert(check.unitreal(10.3) == false);
            assert(check.unitreal(-4) == false);
            assert(check.unitreal(7) == false);
            assert(check.unitreal(0.3) == true);
            assert(check.unitreal('hello') == false);
            assert(check.unitreal(@()fprintf('hello')) == false);
            assert(check.unitreal(struct('hello',10,'world',20)) == false);
            assert(check.unitreal({'0' '1' '2'}) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "string".\n');
            
            assert(check.string(true) == false);
            assert(check.string(false) == false);
            assert(check.string(10.3) == false);
            assert(check.string(-4) == false);
            assert(check.string(7) == false);
            assert(check.string(0.3) == false);
            assert(check.string('hello') == true);
            assert(check.string(@()fprintf('hello')) == false);
            assert(check.string(struct('hello',10,'world',20)) == false);
            assert(check.string({'0' '1' '2'}) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "function_h".\n');
            
            assert(check.function_h(true) == false);
            assert(check.function_h(false) == false);
            assert(check.function_h(10.3) == false);
            assert(check.function_h(-4) == false);
            assert(check.function_h(7) == false);
            assert(check.function_h(0.3) == false);
            assert(check.function_h('hello') == false);
            assert(check.function_h(@()fprintf('hello')) == true);
            assert(check.function_h(struct('hello',10,'world',20)) == false);
            assert(check.function_h({'0' '1' '2'}) == false);
             
            clearvars -except test_figure;
            
            fprintf('  Function "struct".\n');
            
            assert(check.struct(true) == false);
            assert(check.struct(false) == false);
            assert(check.struct(10.3) == false);
            assert(check.struct(-4) == false);
            assert(check.struct(7) == false);
            assert(check.struct(0.3) == false);
            assert(check.struct('hello') == false);
            assert(check.struct(@()fprintf('hello')) == false);
            assert(check.struct(struct('hello',10,'world',20)) == true);
            assert(check.struct({'0' '1' '2'}) == false);
             
            clearvars -except test_figure;
            
            fprintf('  Function "cell".\n');
            
            assert(check.cell(true) == false);
            assert(check.cell(false) == false);
            assert(check.cell(10.3) == false);
            assert(check.cell(-4) == false);
            assert(check.cell(7) == false);
            assert(check.cell(0.3) == false);
            assert(check.cell('hello') == false);
            assert(check.cell(@()fprintf('hello')) == false);
            assert(check.cell(struct('hello',10,'world',20)) == false);
            assert(check.cell({'0' '1' '2'}) == true);
            
            clearvars -except test_figure;
            
            % Tests for functions that check for specific structure.
            
            fprintf('  Function "empty".\n');
            
            assert(check.empty([]) == true);
            assert(check.empty(zeros(1,0)) == true);
            assert(check.empty(zeros(0,1)) == true);
            assert(check.empty('') == true);
            assert(check.empty({}) == true);
            assert(check.empty(1) == false);
            assert(check.empty([1 1 1]) == false);
            assert(check.empty([1;1;1]) == false);
            assert(check.empty(ones(4,3)) == false);
            assert(check.empty(ones(4,3,4)) == false);
            assert(check.empty({1}) == false);
            assert(check.empty({1 1 1}) == false);
            assert(check.empty({1;1;1}) == false);
            assert(check.empty(num2cell(ones(4,3))) == false);
            assert(check.empty(num2cell(ones(4,3,4))) == false);
            assert(check.empty('hello') == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "scalar".\n');
            
            assert(check.scalar([]) == false);
            assert(check.scalar(zeros(1,0)) == false);
            assert(check.scalar(zeros(0,1)) == false);
            assert(check.scalar('') == false);
            assert(check.scalar({}) == false);
            assert(check.scalar(1) == true);
            assert(check.scalar([1 1 1]) == false);
            assert(check.scalar([1;1;1]) == false);
            assert(check.scalar(ones(4,3)) == false);
            assert(check.scalar(ones(4,3,4)) == false);
            assert(check.scalar({1}) == true);
            assert(check.scalar({1 1 1}) == false);
            assert(check.scalar({1;1;1}) == false);
            assert(check.scalar(num2cell(ones(4,3))) == false);
            assert(check.scalar(num2cell(ones(4,3,4))) == false);
            assert(check.scalar('hello') == true);
            
            clearvars -except test_figure;
            
            fprintf('  Function "vector".\n');
            
            assert(check.vector([]) == false);
            assert(check.vector(zeros(1,0)) == false);
            assert(check.vector(zeros(0,1)) == false);
            assert(check.vector('') == false);
            assert(check.vector({}) == false);
            assert(check.vector(1) == true);
            assert(check.vector([1 1 1]) == true);
            assert(check.vector([1;1;1]) == true);
            assert(check.vector(ones(4,3)) == false);
            assert(check.vector(ones(4,3,4)) == false);
            assert(check.vector({1}) == true);
            assert(check.vector({1 1 1}) == true);
            assert(check.vector({1;1;1}) == true);
            assert(check.vector(num2cell(ones(4,3))) == false);
            assert(check.vector(num2cell(ones(4,3,4))) == false);
            assert(check.vector('hello') == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "matrix".\n');
            
            assert(check.matrix([]) == false);
            assert(check.matrix(zeros(1,0)) == false);
            assert(check.matrix(zeros(0,1)) == false);
            assert(check.matrix('') == false);
            assert(check.matrix({}) == false);
            assert(check.matrix(1) == true);
            assert(check.matrix([1 1 1]) == true);
            assert(check.matrix([1;1;1]) == true);
            assert(check.matrix(ones(4,3)) == true);
            assert(check.matrix(ones(4,3,4)) == false);
            assert(check.matrix({1}) == true);
            assert(check.matrix({1 1 1}) == true);
            assert(check.matrix({1;1;1}) == true);
            assert(check.matrix(num2cell(ones(4,3))) == true);
            assert(check.matrix(num2cell(ones(4,3,4))) == false);
            assert(check.matrix('hello') == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "tensor".\n');
            
            assert(check.tensor([]) == false);
            assert(check.tensor(zeros(1,0)) == false);
            assert(check.tensor(zeros(0,1)) == false);
            assert(check.tensor('') == false);
            assert(check.tensor({}) == false);
            assert(check.tensor(1) == true);
            assert(check.tensor([1 1 1]) == true);
            assert(check.tensor([1;1;1]) == true);
            assert(check.tensor(ones(4,3)) == true);
            assert(check.tensor(ones(4,3,4)) == true);
            assert(check.tensor({1}) == true);
            assert(check.tensor({1 1 1}) == true);
            assert(check.tensor({1;1;1}) == true);
            assert(check.tensor(num2cell(ones(4,3))) == true);
            assert(check.tensor(num2cell(ones(4,3,4))) == true);
            assert(check.tensor('hello') == true);
            assert(check.tensor([],3) == false);
            assert(check.tensor(zeros(1,0),3) == false);
            assert(check.tensor(zeros(0,1),3) == false);
            assert(check.tensor('',3) == false);
            assert(check.tensor({},3) == false);
            assert(check.tensor(1,3) == true);
            assert(check.tensor([1 1 1],3) == true);
            assert(check.tensor([1;1;1],3) == true);
            assert(check.tensor(ones(4,3),3) == true);
            assert(check.tensor(ones(4,3,4),3) == true);
            assert(check.tensor({1},3) == true);
            assert(check.tensor({1 1 1},3) == true);
            assert(check.tensor({1;1;1},3) == true);
            assert(check.tensor(num2cell(ones(4,3)),3) == true);
            assert(check.tensor(num2cell(ones(4,3,4)),3) == true);
            assert(check.tensor('hello',0) == false);
            assert(check.tensor('hello',1) == true);
            assert(check.tensor('hello',2) == false);
            assert(check.tensor('hello',3) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "match_dims".\n');
            
            assert(check.match_dims(1,1) == true);
            assert(check.match_dims([1 2 3],[1 3 2]) == true);
            assert(check.match_dims([1;2;3],[1 3 2]) == true);
            assert(check.match_dims(zeros(4,3),zeros(4,3)) == true);
            assert(check.match_dims(zeros(4,4),zeros(4,3)) == true);
            assert(check.match_dims(1,[1 2]) == false);
            assert(check.match_dims([1 2 3],[1 3 2 4]) == false);
            assert(check.match_dims([1;2;3],[1 3 2 4]) == false);
            assert(check.match_dims(zeros(4,2),zeros(3,2)) == false);
            assert(check.match_dims(1,1,1) == true);
            assert(check.match_dims(1,1,2) == true);
            assert(check.match_dims(1,1,3) == true);
            assert(check.match_dims([1 2 3],[1 3 2],1) == true);
            assert(check.match_dims([1 2 3],[1 3 2],2) == true);
            assert(check.match_dims([1 2 3],[1 3 2],3) == true);
            assert(check.match_dims([1 2 3],zeros(3,4),1) == true);
            assert(check.match_dims([1 2 3 4],zeros(3,4),2) == true);
            assert(check.match_dims(zeros(3,4),[1 2 3],1) == true);
            assert(check.match_dims(zeros(3,4),[1 2 3 4],2) == true);
            assert(check.match_dims(zeros(3,3),zeros(3,4),1) == true);
            assert(check.match_dims(zeros(3,4),zeros(2,4),2) == true);
            assert(check.match_dims(1,[1 2],2) == false);
            assert(check.match_dims([1 2 3],[1 3 2 4],2) == false);
            assert(check.match_dims([1 2 3],zeros(3,4),2) == false);
            assert(check.match_dims(zeros(3,4),zeros(4,3),1) == false);
            assert(check.match_dims(zeros(3,4),zeros(4,3),2) == false);
            assert(check.match_dims(1,1,1,1) == true);
            assert(check.match_dims([1 2 3],[1 2 3],1,1) == true);
            assert(check.match_dims([1;2;3],[1 2 3],1,2) == true);
            assert(check.match_dims([1 2 3],[1 2 3],1,3) == true);
            assert(check.match_dims([1;2;3],[1 2 3],1,2) == true);
            assert(check.match_dims(zeros(3,4),zeros(5,3),1,2) == true);
            assert(check.match_dims(zeros(3,4),zeros(3,5),1,1) == true);
            assert(check.match_dims(1,[1 2],1,2) == false);
            assert(check.match_dims([1;2;3],[1 2 3],1,1) == false);
            assert(check.match_dims(zeros(4,3),zeros(4,3),1,2) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "check".\n');
            
            fprintf('    With empty array not good (default).\n');
            
            assert(check.checkv(true) == true);
            assert(check.checkv(false) == false);
            assert(check.checkv(7.3) == true);
            assert(check.checkv(-5) == true);
            assert(check.checkv(4) == true);
            assert(check.checkv(0) == false);
            assert(check.checkv(10) == true);
            assert(check.checkv('hello') == true);
            assert(check.checkv(true(4,2)) == true);
            assert(check.checkv(false(5,3)) == false);
            assert(check.checkv(7.3*ones(4,9)) == true);
            assert(check.checkv(-5*ones(3,2)) == true);
            assert(check.checkv(4*ones(4,4)) == true);
            assert(check.checkv(['hello';'world']) == true);
            assert(check.checkv([1 2 3]) == true);
            assert(check.checkv([0 0 0]) == false);
            assert(check.checkv(zeros(4,4,4) > -1 & zeros(4,4,4) < 1) == true);
            assert(check.checkv([]) == false);
            assert(check.checkv(zeros(1,0)) == false);
            assert(check.checkv(zeros(0,1)) == false);
            assert(check.checkv('') == false);
            assert(check.checkv({}) == false);
            
            clearvars -except test_figure;
            
            fprintf('    With empty array not good.\n');
            
            assert(check.checkv(true,false) == true);
            assert(check.checkv(false,false) == false);
            assert(check.checkv(7.3,false) == true);
            assert(check.checkv(-5,false) == true);
            assert(check.checkv(4,false) == true);
            assert(check.checkv(0,false) == false);
            assert(check.checkv(10,false) == true);
            assert(check.checkv('hello',false) == true);
            assert(check.checkv(true(4,2),false) == true);
            assert(check.checkv(false(5,3),false) == false);
            assert(check.checkv(7.3*ones(4,9),false) == true);
            assert(check.checkv(-5*ones(3,2),false) == true);
            assert(check.checkv(4*ones(4,4),false) == true);
            assert(check.checkv(['hello';'world'],false) == true);
            assert(check.checkv([1 2 3],false) == true);
            assert(check.checkv([0 0 0],false) == false);
            assert(check.checkv(zeros(4,4,4) > -1 & zeros(4,4,4) < 1,false) == true);
            assert(check.checkv([],false) == false);
            assert(check.checkv(zeros(1,0),false) == false);
            assert(check.checkv(zeros(0,1),false) == false);
            assert(check.checkv('',false) == false);
            assert(check.checkv({},false) == false);
            
            clearvars -except test_figure;
            
            fprintf('    With empty array good.\n');
            
            assert(check.checkv(true,true) == true);
            assert(check.checkv(false,true) == false);
            assert(check.checkv(7.3,true) == true);
            assert(check.checkv(-5,true) == true);
            assert(check.checkv(4,true) == true);
            assert(check.checkv(0,true) == false);
            assert(check.checkv(10,true) == true);
            assert(check.checkv('hello',true) == true);
            assert(check.checkv(true(4,2),true) == true);
            assert(check.checkv(false(5,3),true) == false);
            assert(check.checkv(7.3*ones(4,9),true) == true);
            assert(check.checkv(-5*ones(3,2),true) == true);
            assert(check.checkv(4*ones(4,4),true) == true);
            assert(check.checkv(['hello';'world'],true) == true);
            assert(check.checkv([1 2 3],true) == true);
            assert(check.checkv([0 0 0],true) == false);
            assert(check.checkv(zeros(4,4,4) > -1 & zeros(4,4,4) < 1,true) == true);
            assert(check.checkv([],true) == true);
            assert(check.checkv(zeros(1,0),true) == true);
            assert(check.checkv(zeros(0,1),true) == true);
            assert(check.checkv('',true) == true);
            assert(check.checkv({},true) == true);
            
            clearvars -except test_figure;
            
            fprintf('  Function "checkf".\n');
            
            fprintf('    With empty aray not good (default).\n');
            
            assert(check.checkf(@(v)v,[]) == false);
            assert(check.checkf(@(v)v,zeros(1,0)) == false);
            assert(check.checkf(@(v)v,zeros(0,1)) == false);
            assert(check.checkf(@(v)v,'') == false);
            assert(check.checkf(@(v)v,{}) == false);
            assert(check.checkf(@(v)v,true) == true);
            assert(check.checkf(@(v)mod(v,2) == 0,[2 4 6 8]) == true);
            assert(check.checkf(@(v)mod(v,2) == 0,{2 4 6 8}) == true);
            assert(check.checkf(@(v)v,false) == false);
            assert(check.checkf(@(v)mod(v,2) == 0,[2 4 6 7]) == false);
            assert(check.checkf(@(v)mod(v,2) == 0,{2 4 6 7}) == false);
            
            fprintf('    With empty array not good.\n');
            
            assert(check.checkf(@(v)v,[],false) == false);
            assert(check.checkf(@(v)v,zeros(1,0),false) == false);
            assert(check.checkf(@(v)v,zeros(0,1),false) == false);
            assert(check.checkf(@(v)v,'',false) == false);
            assert(check.checkf(@(v)v,{},false) == false);
            assert(check.checkf(@(v)v,true,false) == true);
            assert(check.checkf(@(v)mod(v,2) == 0,[2 4 6 8],false) == true);
            assert(check.checkf(@(v)mod(v,2) == 0,{2 4 6 8},false) == true);
            assert(check.checkf(@(v)v,false,false) == false);
            assert(check.checkf(@(v)mod(v,2) == 0,[2 4 6 7],false) == false);
            assert(check.checkf(@(v)mod(v,2) == 0,{2 4 6 7},false) == false);
            
            fprintf('    With empty aray good.\n');
            
            assert(check.checkf(@(v)v,[],true) == true);
            assert(check.checkf(@(v)v,zeros(1,0),true) == true);
            assert(check.checkf(@(v)v,zeros(0,1),true) == true);
            assert(check.checkf(@(v)v,'',true) == true);
            assert(check.checkf(@(v)v,{},true) == true);
            assert(check.checkf(@(v)v,true,true) == true);
            assert(check.checkf(@(v)mod(v,2) == 0,[2 4 6 8],true) == true);
            assert(check.checkf(@(v)mod(v,2) == 0,{2 4 6 8},true) == true);
            assert(check.checkf(@(v)v,false,true) == false);
            assert(check.checkf(@(v)mod(v,2) == 0,[2 4 6 7],true) == false);
            assert(check.checkf(@(v)mod(v,2) == 0,{2 4 6 7},true) == false);
            
            fprintf('  Function "same".\n');
            
            assert(check.same([],[]) == true);
            assert(check.same([],zeros(1,0)) == true);
            assert(check.same([],zeros(0,1)) == true);
            assert(check.same([],'') == true);
            assert(check.same([],{}) == true);
            assert(check.same(zeros(1,0),[]) == true);
            assert(check.same(zeros(1,0),zeros(1,0)) == true);
            assert(check.same(zeros(1,0),zeros(0,1)) == true);
            assert(check.same(zeros(1,0),'') == true);
            assert(check.same(zeros(1,0),{}) == true);
            assert(check.same(zeros(0,1),[]) == true);
            assert(check.same(zeros(0,1),zeros(1,0)) == true);
            assert(check.same(zeros(0,1),zeros(0,1)) == true);
            assert(check.same(zeros(0,1),'') == true);
            assert(check.same(zeros(0,1),{}) == true);
            assert(check.same('',[]) == true);
            assert(check.same('',zeros(1,0)) == true);
            assert(check.same('',zeros(0,1)) == true);
            assert(check.same('','') == true);
            assert(check.same('',{}) == true);
            assert(check.same({},[]) == true);
            assert(check.same({},zeros(1,0)) == true);
            assert(check.same({},zeros(0,1)) == true);
            assert(check.same({},'') == true);
            assert(check.same({},{}) == true);
            assert(check.same(true,true) == true);
            assert(check.same([true true],[true true]) == true);
            assert(check.same(true,[true false true]) == false);
            assert(check.same([true false],[true false; false true]) == false);
            assert(check.same(true,4.3) == false);
            assert(check.same(true,-4) == false);
            assert(check.same(true,3) == false);
            assert(check.same(true,3.2) == false);
            assert(check.same(true,'hello') == false);
            assert(check.same(false,@check.same) == false);
            assert(check.same(false,struct('hello',false)) == false);
            assert(check.same(false,1) == false);
            assert(check.same(false,{'hello'}) == false);
            assert(check.same(false,{false}) == false);
            assert(check.same(false,{10}) == false);
            assert(check.same(true,false) == false);
            assert(check.same([true false],[true true]) == false);
            assert(check.same(10,10) == true);
            assert(check.same([10 -3 7; 2 1 3.2],[10 -3 7; 2 1 3.2]) == true);
            assert(check.same(10,[4 3 2]) == false);
            assert(check.same([1 2; 3 2],rand(3,3)) == false);
            assert(check.same(5,false) == false);
            assert(check.same(4.2,'hello') == false);
            assert(check.same(-4,@check.same) == false);
            assert(check.same(4,struct('hello',4)) == false);
            assert(check.same(3,{'hello'}) == false);
            assert(check.same(5,{5}) == false);
            assert(check.same(10,7) == false);
            assert(check.same(rand(100,100),rand(100,100)) == false);
            assert(check.same('hello','hello') == true);
            assert(check.same('hello',true) == false);
            assert(check.same('hello',10.3) == false);
            assert(check.same('hello',-10) == false);
            assert(check.same('hello',10) == false);
            assert(check.same('hello',0.3) == false);
            assert(check.same('hello',@check.same) == false);
            assert(check.same('hello',struct('hello','hello')) == false);
            assert(check.same('hello',{'hello'}) == false);
            assert(check.same('hello',{1}) == false);
            assert(check.same('hello','world') == false);
            assert(check.same(@check.same,@check.same) == true);
            assert(check.same(@check.same,{@check.same @check.same @check.same}) == false);
            assert(check.same(@check.same,true) == false);
            assert(check.same(@check.same,10.3) == false);
            assert(check.same(@check.same,-10) == false);
            assert(check.same(@check.same,10) == false);
            assert(check.same(@check.same,0.3) == false);
            assert(check.same(@check.same,'check.same') == false);
            assert(check.same(@check.same,struct('hello',@check.same)) == false);
            assert(check.same(@check.same,{'check.same'}) == false);
            assert(check.same(@check.same,{1}) == false);
            assert(check.same(@check.same,@check.any) == false);
            assert(check.same(struct('hello','hello'),struct('hello','hello')) == true);
            assert(check.same(struct('a',10,'b',20),struct('b',20,'a',10)) == true);
            assert(check.same([struct('a',10,'b',20) struct('b',200,'a',100)],[struct('b',20,'a',10) struct('a',100,'b',200)]) == true);
            assert(check.same(struct('hello',false),false) == false);
            assert(check.same(struct('hello',10.3),10.3) == false);
            assert(check.same(struct('hello',-10),-10) == false);
            assert(check.same(struct('hello',10),10) == false);
            assert(check.same(struct('hello',0.3),0.3) == false);
            assert(check.same(struct('hello','hello'),'hello') == false);
            assert(check.same(struct('hello',@check.same),@check.same) == false);
            assert(check.same(struct('hello',{'hello'}),{'hello'}) == false);
            assert(check.same([struct('hello',{1}) struct('hello',{2}) struct('hello',{3})],{1 2 3}) == false);
            assert(check.same([struct('a',10,'b',20) struct('b',200,'a',100)],[struct('b',20,'a',10) struct('a',100,'b',100)]) == false);
            assert(check.same({'hello'},{'hello'}) == true);
            assert(check.same({'hello' 'world'},{'hello' 'world'}) == true);
            assert(check.same({'hello'},{'hello' 'world'}) == false);
            assert(check.same({'hello'},true) == false);
            assert(check.same({'hello'},10.3) == false);
            assert(check.same({'hello'},-10) == false);
            assert(check.same({'hello'},10) == false);
            assert(check.same({'hello'},0.3) == false);
            assert(check.same({'hello'},'hello') == false);
            assert(check.same({'hello'},@check.same) == false);
            assert(check.same({'hello'},struct('hello',{'hello'})) == false);
            assert(check.same({'hello'},{'world'}) == false);
            assert(check.same({'hello' 'world'},{'world' 'hello'}) == false);
            assert(check.same({1},{1}) == true);
            assert(check.same({1 2 3},{1 2 3}) == true);
            assert(check.same({@check.same @check.any},{@check.same @check.any}) == true);
            assert(check.same({{1 2 3} {'hello' 'world'}},{{1 2 3} {'hello' 'world'}}) == true);
            assert(check.same({1 2},{1 2 3 4}) == false);
            assert(check.same({false},false) == false);
            assert(check.same({10.3},10.3) == false);
            assert(check.same({-10},-10) == false);
            assert(check.same({10},10) == false);
            assert(check.same({0.3},0.3) == false);
            assert(check.same({'hello'},'hello') == false);
            assert(check.same({@check.same},@check.same) == false);
            assert(check.same({struct('hello','hello')},struct('hello','hello')) == false);
            assert(check.same({{'hello'}},{'hello'}) == false);
            t = rand(100,100);
            assert(check.same(1,1) == true);
            assert(check.same(1,1 + 1e-7) == true);
            assert(check.same(t,sqrt(t .^ 2)) == true);
            assert(check.same(t,(t - repmat(mean(t,1),100,1)) + repmat(mean(t,1),100,1)) == true);
            assert(check.same(1,1.5,0.9) == true);
            assert(check.same(2,2.5) == false);
            assert(check.same(1,1 + 1e-7,1e-9) == false);
            
            clearvars -except test_figure;
            
            fprintf('  Function "one_of".\n');
            
            assert(check.one_of(1,1,2,3) == true);
            assert(check.one_of('hello','world','hello') == true);
            assert(check.one_of({1 2},{'hello' 'world'},{rand(100,100) 10},{1 2}) == true);
            assert(check.one_of(4,1,2,3) == false);
            assert(check.one_of('hello','Hello','World') == false);
        end
    end
end
