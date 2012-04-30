classdef tc
    methods (Static,Access=public)
        function [o] = any(obj,classes)
            if exist('classes','var')
                o = tc.atomic(obj,classes) || tc.object(obj);
            else
                o = tc.atomic(obj) || tc.object(obj);
            end
        end
        
        function [o] = atomic(obj,classes)
            if exist('classes','var')
                o = tc.value(obj) || tc.labels(obj) || tc.labels_idx(obj,classes) || tc.cell(obj) || tc.struct(obj);
            else
                o = tc.value(obj) || tc.labels(obj) || tc.cell(obj) || tc.struct(obj);
            end
        end
        
        function [o] = value(obj)
            o = tc.logical(obj) || tc.number(obj) || tc.string(obj) || tc.function_h(obj);
        end
        
        function [o] = logical(obj)
            o = islogical(obj);
        end
        
        function [o] = number(obj)
            o = isnumeric(obj);
        end
        
        function [o] = integer(obj)
            o = isnumeric(obj) && tc.check(floor(obj) == obj);
        end
        
        function [o] = natural(obj)
            o = isnumeric(obj) && tc.check(floor(obj) == obj) && tc.check(obj >= 0);
        end
        
        function [o] = unitreal(obj)
            o = isfloat(obj) && tc.check(obj >= 0 & obj <= 1);
        end
        
        function [o] = string(obj)
            o = ischar(obj);
        end
        
        function [o] = function_h(obj)
            o = isa(obj,'function_handle');
        end
        
        function [o] = labels(obj)
            o = tc.cell(obj) && all(cellfun(@(c)tc.scalar(c) && tc.string(c),obj));
        end
        
        function [o] = labels_idx(obj,classes)
            o = tc.natural(obj) && tc.check(obj >= 1 & obj <= length(classes));
        end
        
        function [o] = cell(obj)
            o = iscell(obj);
        end
        
        function [o] = struct(obj,varargin)
            o = isstruct(obj) && (tc.empty(varargin) || all(isfield(obj,varargin)));
        end
        
        function [o] = object(obj)
            o = tc.dataset(obj) || tc.classification_info(obj) || ...
                tc.experiment(obj) || tc.architecture(obj) || tc.transform(obj) || tc.classifier(obj) || ...
                tc.logging_logger(obj) || tc.logging_handler(obj) || tc.logging_level(obj);
        end
        
        function [o] = dataset(obj)
            o = tc.dataset_record(obj) || tc.dataset_image(obj);
        end
        
        function [o] = dataset_record(obj)
            o = tc.matrix(obj) && tc.number(obj);
        end
        
        function [o] = dataset_image(obj)
            o = tc.tensor(obj,4) && tc.number(obj);
        end
        
        function [o] = classification_info(obj)
            o = isa(obj,'classification_info');
        end
        
        function [o] = experiment(obj)
            o = isa(obj,'experiment');
        end
        
        function [o] = architecture(obj)
            o = isa(obj,'architecture');
        end
        
        function [o] = transform(obj)
            o = isa(obj,'transform');
        end
        
        function [o] = transforms_reversible(obj)
            o = isa(obj,'transforms.reversible');
        end
        
        function [o] = classifier(obj)
            o = isa(obj,'classifier');
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
            if tc.string(obj)
                o = (length(size(obj)) == 2) && ((size(obj,1) == 1) && (size(obj,2) >= 1));
            else
                o = (length(size(obj)) == 2) && (size(obj,1) == 1) && (size(obj,2) == 1);
            end
        end
        
        function [o] = vector(obj)
            if tc.string(obj)
                o = false;
            else
                o = (length(size(obj)) == 2) && (((size(obj,1) == 1) && (size(obj,2) >= 1)) || ...
                                               ((size(obj,1) >= 1) && (size(obj,2) == 1)));
            end
        end
        
        function [o] = matrix(obj)
            if tc.string(obj)
                o = false;
            else
                o = (length(size(obj)) == 2) && (size(obj,1) >= 1) && (size(obj,2) >= 1);
            end
        end
        
        function [o] = tensor(obj,d)
            if ~exist('d','var')
                o = ~tc.empty(obj);
            else
                if tc.string(obj)
                    if d == 1
                        o = true;
                    else
                        o = false;
                    end
                else
                    if d == 0
                        o = tc.scalar(obj);
                    elseif d == 1
                        o = tc.vector(obj);
                    elseif d == 2
                        o = tc.matrix(obj);
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
                if tc.vector(obj1) && tc.vector(obj2)
                    o = size(obj1,d1) == size(obj2,d1);
                elseif tc.vector(obj1)
                    o = length(obj1) == size(obj2,d1);
                elseif tc.vector(obj2)
                    o = length(obj2) == size(obj1,d1);
                else
                    o = size(obj1,d1) == size(obj2,d1);
                end
            else
                if tc.vector(obj1) && tc.vector(obj2);
                    o = length(obj1) == length(obj2);
                else
                    o = size(obj1,1) == size(obj2,1);
                end
            end
        end
                
        function [o] = check(obj,empty_good)
            if exist('empty_good','var')
                empty_good_t = empty_good;
            else
                empty_good_t = false;
            end
            
            if ~tc.empty(obj)
                o_t = all(obj(:));
            else
                o_t = false;
            end
            
            if empty_good_t
                o = tc.empty(obj) || o_t;
            else
                o = ~tc.empty(obj) && o_t;
            end
        end
        
        function [o] = checkf(check_fn,obj,empty_good)
            if exist('empty_good','var')
                empty_good_t = empty_good;
            else
                empty_good_t = false;
            end
            
            if tc.cell(obj)
                o_t = all(cellfun(check_fn,obj(:)));
            else
                o_t = all(arrayfun(check_fn,obj(:)));
            end
            
            if empty_good_t
                o = tc.empty(obj) || o_t;
            else
                o = ~tc.empty(obj) && o_t;
            end
        end
        
        function [o] = same(obj1,obj2,varargin)
            if isempty(varargin)
                has_classes = false;
                classes = -1;
                approx_epsilon = 1e-6;
            elseif length(varargin) == 2
                if strcmp(varargin{1},'Classes')
                    assert(tc.vector(varargin{2}));
                    assert(tc.labels(varargin{2}));
                    
                    has_classes = true;
                    classes = varargin{2};
                    approx_epsilon = 1e-6;
                elseif strcmp(varargin{1},'Epsilon')
                    assert(tc.scalar(varargin{2}));
                    assert(tc.unitreal(varargin{2}));
                    
                    has_classes = false;
                    classes = -1;
                    approx_epsilon = varargin{2};
                else
                    assert(false);
                end
            else
                assert(false);
            end
            
            if tc.empty(obj1) && tc.empty(obj2)
                o = true;
            else
                assert((tc.tensor(obj1) && tc.logical(obj1)) || ...
                       (tc.tensor(obj1) && tc.number(obj1)) || ...
                       (tc.scalar(obj1) && tc.string(obj1)) || ...
                       (tc.scalar(obj1) && tc.function_h(obj1)) || ...
                       (tc.vector(obj1) && tc.labels(obj1)) || ...
                       (has_classes && tc.vector(obj1) && tc.labels_idx(obj1,classes)) || ...
                       (tc.vector(obj1) && tc.cell(obj1)) || ...
                       (tc.vector(obj1) && tc.struct(obj1)) || ...
                       (tc.vector(obj1) && tc.object(obj1)));
                assert((tc.tensor(obj2) && tc.logical(obj2)) || ...
                       (tc.tensor(obj2) && tc.number(obj2)) || ...
                       (tc.scalar(obj2) && tc.string(obj2)) || ...
                       (tc.scalar(obj2) && tc.function_h(obj2)) || ...
                       (tc.vector(obj2) && tc.labels(obj2)) || ...
                       (has_classes && tc.vector(obj2) && tc.labels_idx(obj2,classes)) || ...
                       (tc.vector(obj2) && tc.cell(obj2)) || ...
                       (tc.vector(obj2) && tc.struct(obj2)) || ...
                       (tc.vector(obj2) && tc.object(obj2)));

                o = true;
                o = o && tc.check(size(obj1) == size(obj2));
                o = o && ((tc.logical(obj1) && tc.logical(obj2)) || ...
                          (tc.number(obj1) && tc.number(obj2)) || ...
                          (tc.string(obj1) && tc.string(obj2)) || ...
                          (tc.function_h(obj1) && tc.function_h(obj2)) || ...
                          (tc.labels(obj1) && tc.labels(obj2)) || ...
                          (has_classes && tc.labels_idx(obj1,classes) && tc.labels_idx(obj2,classes)) || ...
                          (tc.cell(obj1) && tc.cell(obj2)) || ...
                          (tc.struct(obj1) && tc.struct(obj2)) || ...
                          (tc.object(obj1) && tc.object(obj2)));

                if o
                    if has_classes && tc.labels_idx(obj1,classes)
                        o = tc.check(obj1 == obj2);
                    elseif tc.logical(obj1)
                        o = tc.check(obj1 == obj2);
                    elseif tc.number(obj1)
                        o = tc.check(abs(obj1 - obj2) < approx_epsilon);
                    elseif tc.string(obj1)
                        o = strcmp(obj1,obj2);
                    elseif tc.function_h(obj1)
                        o = strcmp(func2str(obj1),func2str(obj2));
                    elseif tc.labels(obj1)
                        o = tc.check(cellfun(@strcmp,obj1,obj2));
                    elseif tc.cell(obj1)
                        o = tc.check(cellfun(@tc.same,obj1,obj2));
                    elseif tc.struct(obj1)
                        obj1p = orderfields(obj1);
                        obj2p = orderfields(obj2);

                        f1 = fieldnames(obj1p);
                        f2 = fieldnames(obj2p);

                        o = tc.same(f1,f2);

                        if o
                            o = tc.check(arrayfun(@(s1,s2)tc.check(cellfun(@tc.same,struct2cell(s1),struct2cell(s2))),obj1p,obj2p));
                        end
                    elseif tc.object(obj1)
                        o = tc.check(arrayfun(@(ii)obj1(ii) == obj2(ii),1:length(obj1)));
                    else
                        assert(false);
                    end
                end
            end
        end
        
        function [o] = one_of(obj,varargin)
            o = ~tc.empty(find(cellfun(@(c)tc.same(obj,c),varargin),1));
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
            assert(tc.any(@()fprintf('hello')) == true);
            assert(tc.any({'0' '1' '2'}) == true);
            assert(tc.any([1 2 3],{'0' '1' '2'}) == true);
            assert(tc.any({0}) == true);
            assert(tc.any({1 2 3}) == true);
            assert(tc.any({'hello' 'world'}) == true);
            assert(tc.any(struct('hello','world')) == true);
            
            clearvars -except display;
            
            fprintf('  Function "atomic".\n');
            
            assert(tc.atomic(true) == true);
            assert(tc.atomic(false) == true);
            assert(tc.atomic(10.3) == true);
            assert(tc.atomic(-4) == true);
            assert(tc.atomic(7) == true);
            assert(tc.atomic(0.3) == true);
            assert(tc.atomic('hello') == true);
            assert(tc.atomic(@()fprintf('hello')) == true);
            assert(tc.atomic({'0' '1' '2'}) == true);
            assert(tc.atomic([1 2 3],{'0' '1' '2'}) == true);
            assert(tc.atomic({0}) == true);
            assert(tc.atomic({1 2 3}) == true);
            assert(tc.atomic({'hello' 'world'}) == true);
            assert(tc.atomic(struct('hello','world')) == true);
            
            clearvars -except display;
            
            fprintf('  Function "value".\n');
            
            assert(tc.value(true) == true);
            assert(tc.value(false) == true);
            assert(tc.value(10.3) == true);
            assert(tc.value(-4) == true);
            assert(tc.value(7) == true);
            assert(tc.value(0.3) == true);
            assert(tc.value('hello') == true);
            assert(tc.value(@()fprintf('hello')) == true);
            assert(tc.value({'0' '1' '2'}) == false);
            assert(tc.value({0}) == false);
            assert(tc.value({1 2 3}) == false);
            assert(tc.value({'hello' 'world'}) == false);
            assert(tc.value(struct('hello','world')) == false);
            
            clearvars -except display;
            
            fprintf('  Function "logical".\n');
            
            assert(tc.logical(true) == true);
            assert(tc.logical(false) == true);
            assert(tc.logical(10.3) == false);
            assert(tc.logical(-4) == false);
            assert(tc.logical(7) == false);
            assert(tc.logical(0.3) == false);
            assert(tc.logical('hello') == false);
            assert(tc.logical(@()fprintf('hello')) == false);
            assert(tc.logical({'0' '1' '2'}) == false);
            assert(tc.logical({0}) == false);
            assert(tc.logical({1 2 3}) == false);
            assert(tc.logical({'hello' 'world'}) == false);
            assert(tc.logical(struct('hello','world')) == false);
            
            clearvars -except display;
            
            fprintf('  Function "number".\n');
            
            assert(tc.number(true) == false);
            assert(tc.number(false) == false);
            assert(tc.number(10.3) == true);
            assert(tc.number(-4) == true);
            assert(tc.number(7) == true);
            assert(tc.number(0.3) == true);
            assert(tc.number('hello') == false);
            assert(tc.number(@()fprintf('hello')) == false);
            assert(tc.number({'0' '1' '2'}) == false);
            assert(tc.number({0}) == false);
            assert(tc.number({1 2 3}) == false);
            assert(tc.number({'hello' 'world'}) == false);
            assert(tc.number(struct('hello','world')) == false);
            
            clearvars -except display;
            
            fprintf('  Function "integer".\n');
            
            assert(tc.integer(true) == false);
            assert(tc.integer(false) == false);
            assert(tc.integer(10.3) == false);
            assert(tc.integer(-4) == true);
            assert(tc.integer(7) == true);
            assert(tc.integer(0.3) == false);
            assert(tc.integer('hello') == false);
            assert(tc.integer(@()fprintf('hello')) == false);
            assert(tc.integer({'0' '1' '2'}) == false);
            assert(tc.integer({0}) == false);
            assert(tc.integer({1 2 3}) == false);
            assert(tc.integer({'hello' 'world'}) == false);
            assert(tc.integer(struct('hello','world')) == false);
            
            clearvars -except display;
            
            fprintf('  Function "natural".\n');
            
            assert(tc.natural(true) == false);
            assert(tc.natural(false) == false);
            assert(tc.natural(10.3) == false);
            assert(tc.natural(-4) == false);
            assert(tc.natural(7) == true);
            assert(tc.natural(0.3) == false);
            assert(tc.natural('hello') == false);
            assert(tc.natural(@()fprintf('hello')) == false);
            assert(tc.natural({'0' '1' '2'}) == false);
            assert(tc.natural({0}) == false);
            assert(tc.natural({1 2 3}) == false);
            assert(tc.natural({'hello' 'world'}) == false);
            assert(tc.natural(struct('hello','world')) == false);
            
            clearvars -except display;
            
            fprintf('  Function "unitreal".\n');
            
            assert(tc.unitreal(true) == false);
            assert(tc.unitreal(false) == false);
            assert(tc.unitreal(10.3) == false);
            assert(tc.unitreal(-4) == false);
            assert(tc.unitreal(7) == false);
            assert(tc.unitreal(0.3) == true);
            assert(tc.unitreal('hello') == false);
            assert(tc.unitreal(@()fprintf('hello')) == false);
            assert(tc.unitreal({'0' '1' '2'}) == false);
            assert(tc.unitreal({0}) == false);
            assert(tc.unitreal({1 2 3}) == false);
            assert(tc.unitreal({'hello' 'world'}) == false);
            assert(tc.unitreal(struct('hello','world')) == false);
            
            clearvars -except display;
            
            fprintf('  Function "string".\n');
            
            assert(tc.string(true) == false);
            assert(tc.string(false) == false);
            assert(tc.string(10.3) == false);
            assert(tc.string(-4) == false);
            assert(tc.string(7) == false);
            assert(tc.string(0.3) == false);
            assert(tc.string('hello') == true);
            assert(tc.string(@()fprintf('hello')) == false);
            assert(tc.string({'0' '1' '2'}) == false);
            assert(tc.string({0}) == false);
            assert(tc.string({1 2 3}) == false);
            assert(tc.string({'hello' 'world'}) == false);
            assert(tc.string(struct('hello','world')) == false);
            
            clearvars -except display;
            
            fprintf('  Function "function_h".\n');
            
            assert(tc.function_h(true) == false);
            assert(tc.function_h(false) == false);
            assert(tc.function_h(10.3) == false);
            assert(tc.function_h(-4) == false);
            assert(tc.function_h(7) == false);
            assert(tc.function_h(0.3) == false);
            assert(tc.function_h('hello') == false);
            assert(tc.function_h(@()fprintf('hello')) == true);
            assert(tc.function_h({'0' '1' '2'}) == false);
            assert(tc.function_h({0}) == false);
            assert(tc.function_h({1 2 3}) == false);
            assert(tc.function_h({'hello' 'world'}) == false);
            assert(tc.function_h(struct('hello','world')) == false);
             
            clearvars -except display;
            
            fprintf('  Function "labels".\n');
            
            assert(tc.labels(true) == false);
            assert(tc.labels(false) == false);
            assert(tc.labels(10.3) == false);
            assert(tc.labels(-4) == false);
            assert(tc.labels(7) == false);
            assert(tc.labels(0.3) == false);
            assert(tc.labels('hello') == false);
            assert(tc.labels(@()fprintf('hello')) == false);
            assert(tc.labels({'0' '1' '2'}) == true);
            assert(tc.labels({0}) == false);
            assert(tc.labels({1 2 3}) == false);
            assert(tc.labels({'hello' 'world'}) == true);
            assert(tc.labels(struct('hello','world')) == false);
            
            clearvars -except display;
            
            fprintf('  Function "labels_idx".\n');
            
            assert(tc.labels_idx(true,{'0' '1'}) == false);
            assert(tc.labels_idx(false,{'0' '1'}) == false);
            assert(tc.labels_idx(10.3,{'0' '1'}) == false);
            assert(tc.labels_idx(-4,{'0' '1'}) == false);
            assert(tc.labels_idx(7,{'0' '1'}) == false);
            assert(tc.labels_idx(0.3,{'0' '1'}) == false);
            assert(tc.labels_idx('hello',{'0' '1'}) == false);
            assert(tc.labels_idx(@()fprintf('hello'),{'0' '1'}) == false);
            assert(tc.labels_idx({'0' '1' '2'},{'0' '1'}) == false);
            assert(tc.labels_idx([1 1 2],{'0' '1'}) == true);
            assert(tc.labels_idx({0},{'0' '1'}) == false);
            assert(tc.labels_idx({1 2 3},{'0' '1'}) == false);
            assert(tc.labels_idx({'hello' 'world'},{'0' '1'}) == false);
            assert(tc.labels_idx(struct('hello','world'),{'0' '1'}) == false);
            
            clearvars -except display;
            
            fprintf('  Function "cell".\n');
            
            assert(tc.cell(true) == false);
            assert(tc.cell(false) == false);
            assert(tc.cell(10.3) == false);
            assert(tc.cell(-4) == false);
            assert(tc.cell(7) == false);
            assert(tc.cell(0.3) == false);
            assert(tc.cell('hello') == false);
            assert(tc.cell(@()fprintf('hello')) == false);
            assert(tc.cell({'0' '1' '2'}) == true);
            assert(tc.cell({0}) == true);
            assert(tc.cell({1 2 3}) == true);
            assert(tc.cell({'hello' 'world'}) == true);
            assert(tc.cell(struct('hello','world')) == false);
            
            clearvars -except display;
            
            fprintf('  Function "struct".\n');
            
            assert(tc.struct(true) == false);
            assert(tc.struct(false) == false);
            assert(tc.struct(10.3) == false);
            assert(tc.struct(-4) == false);
            assert(tc.struct(7) == false);
            assert(tc.struct(0.3) == false);
            assert(tc.struct('hello') == false);
            assert(tc.struct(@()fprintf('hello')) == false);
            assert(tc.struct({'0' '1' '2'}) == false);
            assert(tc.struct({0}) == false);
            assert(tc.struct({1 2 3}) == false);
            assert(tc.struct({'hello' 'world'}) == false);
            assert(tc.struct(struct('hello','world')) == true);
            assert(tc.struct(struct('hello','world'),'hello') == true);
            assert(tc.struct(struct('hello','world','hello2','AAA'),'hello') == true);
            assert(tc.struct(struct('hello','world'),'hello','world') == false);
            
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
            assert(tc.empty('hello') == false);
            
            clearvars -except display;
            
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
            assert(tc.scalar('hello') == true);
            
            clearvars -except display;
            
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
            assert(tc.vector('hello') == false);
            
            clearvars -except display;
            
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
            assert(tc.matrix('hello') == false);
            
            clearvars -except display;
            
            fprintf('  Function "tensor".\n');
            
            assert(tc.tensor([]) == false);
            assert(tc.tensor(zeros(1,0)) == false);
            assert(tc.tensor(zeros(0,1)) == false);
            assert(tc.tensor('') == false);
            assert(tc.tensor({}) == false);
            assert(tc.tensor(1) == true);
            assert(tc.tensor([1 1 1]) == true);
            assert(tc.tensor([1;1;1]) == true);
            assert(tc.tensor(ones(4,3)) == true);
            assert(tc.tensor(ones(4,3,4)) == true);
            assert(tc.tensor({1}) == true);
            assert(tc.tensor({1 1 1}) == true);
            assert(tc.tensor({1;1;1}) == true);
            assert(tc.tensor(num2cell(ones(4,3))) == true);
            assert(tc.tensor(num2cell(ones(4,3,4))) == true);
            assert(tc.tensor('hello') == true);
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
            assert(tc.tensor('hello',0) == false);
            assert(tc.tensor('hello',1) == true);
            assert(tc.tensor('hello',2) == false);
            assert(tc.tensor('hello',3) == false);
            
            clearvars -except display;
            
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
            
            clearvars -except display;
            
            fprintf('  Function "check".\n');
            
            fprintf('    With empty array not good (default).\n');
            
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
            assert(tc.check([]) == false);
            assert(tc.check(zeros(1,0)) == false);
            assert(tc.check(zeros(0,1)) == false);
            assert(tc.check('') == false);
            assert(tc.check({}) == false);
            
            clearvars -except display;
            
            fprintf('    With empty array not good.\n');
            
            assert(tc.check(true,false) == true);
            assert(tc.check(false,false) == false);
            assert(tc.check(7.3,false) == true);
            assert(tc.check(-5,false) == true);
            assert(tc.check(4,false) == true);
            assert(tc.check(0,false) == false);
            assert(tc.check(10,false) == true);
            assert(tc.check('hello',false) == true);
            assert(tc.check(true(4,2),false) == true);
            assert(tc.check(false(5,3),false) == false);
            assert(tc.check(7.3*ones(4,9),false) == true);
            assert(tc.check(-5*ones(3,2),false) == true);
            assert(tc.check(4*ones(4,4),false) == true);
            assert(tc.check(['hello';'world'],false) == true);
            assert(tc.check([1 2 3],false) == true);
            assert(tc.check([0 0 0],false) == false);
            assert(tc.check(zeros(4,4,4) > -1 & zeros(4,4,4) < 1,false) == true);
            assert(tc.check([],false) == false);
            assert(tc.check(zeros(1,0),false) == false);
            assert(tc.check(zeros(0,1),false) == false);
            assert(tc.check('',false) == false);
            assert(tc.check({},false) == false);
            
            clearvars -except display;
            
            fprintf('    With empty array good.\n');
            
            assert(tc.check(true,true) == true);
            assert(tc.check(false,true) == false);
            assert(tc.check(7.3,true) == true);
            assert(tc.check(-5,true) == true);
            assert(tc.check(4,true) == true);
            assert(tc.check(0,true) == false);
            assert(tc.check(10,true) == true);
            assert(tc.check('hello',true) == true);
            assert(tc.check(true(4,2),true) == true);
            assert(tc.check(false(5,3),true) == false);
            assert(tc.check(7.3*ones(4,9),true) == true);
            assert(tc.check(-5*ones(3,2),true) == true);
            assert(tc.check(4*ones(4,4),true) == true);
            assert(tc.check(['hello';'world'],true) == true);
            assert(tc.check([1 2 3],true) == true);
            assert(tc.check([0 0 0],true) == false);
            assert(tc.check(zeros(4,4,4) > -1 & zeros(4,4,4) < 1,true) == true);
            assert(tc.check([],true) == true);
            assert(tc.check(zeros(1,0),true) == true);
            assert(tc.check(zeros(0,1),true) == true);
            assert(tc.check('',true) == true);
            assert(tc.check({},true) == true);
            
            clearvars -except display;
            
            fprintf('  Function "checkf".\n');
            
            fprintf('    With empty aray not good (default).\n');
            
            assert(tc.checkf(@(v)v,[]) == false);
            assert(tc.checkf(@(v)v,zeros(1,0)) == false);
            assert(tc.checkf(@(v)v,zeros(0,1)) == false);
            assert(tc.checkf(@(v)v,'') == false);
            assert(tc.checkf(@(v)v,{}) == false);
            assert(tc.checkf(@(v)v,true) == true);
            assert(tc.checkf(@(v)mod(v,2) == 0,[2 4 6 8]) == true);
            assert(tc.checkf(@(v)mod(v,2) == 0,{2 4 6 8}) == true);
            assert(tc.checkf(@(v)v,false) == false);
            assert(tc.checkf(@(v)mod(v,2) == 0,[2 4 6 7]) == false);
            assert(tc.checkf(@(v)mod(v,2) == 0,{2 4 6 7}) == false);
            
            fprintf('    With empty array not good.\n');
            
            assert(tc.checkf(@(v)v,[],false) == false);
            assert(tc.checkf(@(v)v,zeros(1,0),false) == false);
            assert(tc.checkf(@(v)v,zeros(0,1),false) == false);
            assert(tc.checkf(@(v)v,'',false) == false);
            assert(tc.checkf(@(v)v,{},false) == false);
            assert(tc.checkf(@(v)v,true,false) == true);
            assert(tc.checkf(@(v)mod(v,2) == 0,[2 4 6 8],false) == true);
            assert(tc.checkf(@(v)mod(v,2) == 0,{2 4 6 8},false) == true);
            assert(tc.checkf(@(v)v,false,false) == false);
            assert(tc.checkf(@(v)mod(v,2) == 0,[2 4 6 7],false) == false);
            assert(tc.checkf(@(v)mod(v,2) == 0,{2 4 6 7},false) == false);
            
            fprintf('    With empty aray good.\n');
            
            assert(tc.checkf(@(v)v,[],true) == true);
            assert(tc.checkf(@(v)v,zeros(1,0),true) == true);
            assert(tc.checkf(@(v)v,zeros(0,1),true) == true);
            assert(tc.checkf(@(v)v,'',true) == true);
            assert(tc.checkf(@(v)v,{},true) == true);
            assert(tc.checkf(@(v)v,true,true) == true);
            assert(tc.checkf(@(v)mod(v,2) == 0,[2 4 6 8],true) == true);
            assert(tc.checkf(@(v)mod(v,2) == 0,{2 4 6 8},true) == true);
            assert(tc.checkf(@(v)v,false,true) == false);
            assert(tc.checkf(@(v)mod(v,2) == 0,[2 4 6 7],true) == false);
            assert(tc.checkf(@(v)mod(v,2) == 0,{2 4 6 7},true) == false);
            
            fprintf('  Function "same".\n');
            
            assert(tc.same([],[]) == true);
            assert(tc.same([],zeros(1,0)) == true);
            assert(tc.same([],zeros(0,1)) == true);
            assert(tc.same([],'') == true);
            assert(tc.same([],{}) == true);
            assert(tc.same(zeros(1,0),[]) == true);
            assert(tc.same(zeros(1,0),zeros(1,0)) == true);
            assert(tc.same(zeros(1,0),zeros(0,1)) == true);
            assert(tc.same(zeros(1,0),'') == true);
            assert(tc.same(zeros(1,0),{}) == true);
            assert(tc.same(zeros(0,1),[]) == true);
            assert(tc.same(zeros(0,1),zeros(1,0)) == true);
            assert(tc.same(zeros(0,1),zeros(0,1)) == true);
            assert(tc.same(zeros(0,1),'') == true);
            assert(tc.same(zeros(0,1),{}) == true);
            assert(tc.same('',[]) == true);
            assert(tc.same('',zeros(1,0)) == true);
            assert(tc.same('',zeros(0,1)) == true);
            assert(tc.same('','') == true);
            assert(tc.same('',{}) == true);
            assert(tc.same({},[]) == true);
            assert(tc.same({},zeros(1,0)) == true);
            assert(tc.same({},zeros(0,1)) == true);
            assert(tc.same({},'') == true);
            assert(tc.same({},{}) == true);
            assert(tc.same(true,true) == true);
            assert(tc.same([true true],[true true]) == true);
            assert(tc.same(true,[true false true]) == false);
            assert(tc.same([true false],[true false; false true]) == false);
            assert(tc.same(true,4.3) == false);
            assert(tc.same(true,-4) == false);
            assert(tc.same(true,3) == false);
            assert(tc.same(true,3.2) == false);
            assert(tc.same(true,'hello') == false);
            assert(tc.same(false,@tc.same) == false);
            assert(tc.same(false,1) == false);
            assert(tc.same(false,{'hello'}) == false);
            assert(tc.same(false,{false}) == false);
            assert(tc.same(false,{10}) == false);
            assert(tc.same(false,struct('hello',false)) == false);
            assert(tc.same(true,false) == false);
            assert(tc.same([true false],[true true]) == false);
            assert(tc.same(10,10) == true);
            assert(tc.same([10 -3 7; 2 1 3.2],[10 -3 7; 2 1 3.2]) == true);
            assert(tc.same(10,[4 3 2]) == false);
            assert(tc.same([1 2; 3 2],rand(3,3)) == false);
            assert(tc.same(5,false) == false);
            assert(tc.same(4.2,'hello') == false);
            assert(tc.same(-4,@tc.same) == false);
            assert(tc.same(3,{'hello'}) == false);
            assert(tc.same(5,{5}) == false);
            assert(tc.same(4,struct('hello',4)) == false);
            assert(tc.same(10,7) == false);
            assert(tc.same(rand(100,100),rand(100,100)) == false);
            assert(tc.same('hello','hello') == true);
            assert(tc.same('hello',true) == false);
            assert(tc.same('hello',10.3) == false);
            assert(tc.same('hello',-10) == false);
            assert(tc.same('hello',10) == false);
            assert(tc.same('hello',0.3) == false);
            assert(tc.same('hello',@tc.same) == false);
            assert(tc.same('hello',{'hello'}) == false);
            assert(tc.same('hello',{1}) == false);
            assert(tc.same('hello',struct('hello','hello')) == false);
            assert(tc.same('hello','world') == false);
            assert(tc.same(@tc.same,@tc.same) == true);
            assert(tc.same(@tc.same,{@tc.same @tc.same @tc.same}) == false);
            assert(tc.same(@tc.same,true) == false);
            assert(tc.same(@tc.same,10.3) == false);
            assert(tc.same(@tc.same,-10) == false);
            assert(tc.same(@tc.same,10) == false);
            assert(tc.same(@tc.same,0.3) == false);
            assert(tc.same(@tc.same,'tc.same') == false);
            assert(tc.same(@tc.same,{'tc.same'}) == false);
            assert(tc.same(@tc.same,{1}) == false);
            assert(tc.same(@tc.same,struct('hello',@tc.same)) == false);
            assert(tc.same(@tc.same,@tc.any) == false);
            assert(tc.same({'hello'},{'hello'}) == true);
            assert(tc.same({'hello' 'world'},{'hello' 'world'}) == true);
            assert(tc.same({'hello'},{'hello' 'world'}) == false);
            assert(tc.same({'hello'},true) == false);
            assert(tc.same({'hello'},10.3) == false);
            assert(tc.same({'hello'},-10) == false);
            assert(tc.same({'hello'},10) == false);
            assert(tc.same({'hello'},0.3) == false);
            assert(tc.same({'hello'},'hello') == false);
            assert(tc.same({'hello'},@tc.same) == false);
            assert(tc.same({'hello'},struct('hello',{'hello'})) == false);
            assert(tc.same({'hello'},{'world'}) == false);
            assert(tc.same({'hello' 'world'},{'world' 'hello'}) == false);
            assert(tc.same({1},{1}) == true);
            assert(tc.same({1 2 3},{1 2 3}) == true);
            assert(tc.same({@tc.same @tc.any},{@tc.same @tc.any}) == true);
            assert(tc.same({{1 2 3} {'hello' 'world'}},{{1 2 3} {'hello' 'world'}}) == true);
            assert(tc.same({1 2},{1 2 3 4}) == false);
            assert(tc.same({false},false) == false);
            assert(tc.same({10.3},10.3) == false);
            assert(tc.same({-10},-10) == false);
            assert(tc.same({10},10) == false);
            assert(tc.same({0.3},0.3) == false);
            assert(tc.same({'hello'},'hello') == false);
            assert(tc.same({@tc.same},@tc.same) == false);
            assert(tc.same({{'hello'}},{'hello'}) == false);
            assert(tc.same({struct('hello','hello')},struct('hello','hello')) == false);
            assert(tc.same(struct('hello','hello'),struct('hello','hello')) == true);
            assert(tc.same(struct('a',10,'b',20),struct('b',20,'a',10)) == true);
            assert(tc.same([struct('a',10,'b',20) struct('b',200,'a',100)],[struct('b',20,'a',10) struct('a',100,'b',200)]) == true);
            assert(tc.same(struct('hello',false),false) == false);
            assert(tc.same(struct('hello',10.3),10.3) == false);
            assert(tc.same(struct('hello',-10),-10) == false);
            assert(tc.same(struct('hello',10),10) == false);
            assert(tc.same(struct('hello',0.3),0.3) == false);
            assert(tc.same(struct('hello','hello'),'hello') == false);
            assert(tc.same(struct('hello',@tc.same),@tc.same) == false);
            assert(tc.same(struct('hello',{'hello'}),{'hello'}) == false);
            assert(tc.same([struct('hello',{1}) struct('hello',{2}) struct('hello',{3})],{1 2 3}) == false);
            assert(tc.same([struct('a',10,'b',20) struct('b',200,'a',100)],[struct('b',20,'a',10) struct('a',100,'b',100)]) == false);
            assert(tc.same([1 2 3],[1 2 3],'Classes',{'a' 'b' 'c'}) == true);
            assert(tc.same([1 2 3 4],[1 2 3 4],'Classes',{'a' 'b' 'c'}) == true);
            t = rand(100,100);
            assert(tc.same(1,1) == true);
            assert(tc.same(1,1 + 1e-7) == true);
            assert(tc.same(t,sqrt(t .^ 2)) == true);
            assert(tc.same(t,(t - repmat(mean(t,1),100,1)) + repmat(mean(t,1),100,1)) == true);
            assert(tc.same(1,1.5,'Epsilon',0.9) == true);
            assert(tc.same(2,2.5) == false);
            assert(tc.same(1,1 + 1e-7,'Epsilon',1e-9) == false);
            
            clearvars -except display;
            
            fprintf('  Function "one_of".\n');
            
            assert(tc.one_of(1,1,2,3) == true);
            assert(tc.one_of('hello','world','hello') == true);
            assert(tc.one_of({1 2},{'hello' 'world'},{rand(100,100) 10},{1 2}) == true);
            assert(tc.one_of(4,1,2,3) == false);
            assert(tc.one_of('hello','Hello','World') == false);
        end
    end
end
