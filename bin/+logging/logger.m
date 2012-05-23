classdef logger < handle
    properties (GetAccess=public,SetAccess=immutable)
        handlers;
        handlers_count;
    end
    
    properties (GetAccess=public,SetAccess=private)
        active;
        level_stack;
        per_handler_indent;
        level_at_indent;
    end
    
    methods (Access=public)
        function [obj] = logger(handlers,level_stack,per_handler_indent,level_at_indent)
            assert(tc.vector(handlers));
            assert(tc.cell(handlers));
            assert(tc.checkf(@tc.scalar,handlers));
            assert(tc.checkf(@tc.logging_handler,handlers));
            assert(tc.checkf(@(h)h.active,handlers));
            assert(~exist('level_stack','var') || tc.vector(level_stack));
            assert(~exist('level_stack','var') || tc.logging_level(level_stack));
            assert(~exist('per_handler_indent','var') || tc.vector(per_handler_indent));
            assert(~exist('per_handler_indent','var') || (length(per_handler_indent) == length(handlers)));
            assert(~exist('per_handler_indent','var') || tc.natural(per_handler_indent));
            assert(~exist('level_at_indent','var') || tc.vector(level_at_indent));
            assert(~exist('level_at_indent','var') || (length(level_at_indent) == length(handlers)));
            assert(~exist('level_at_indent','var') || tc.cell(level_at_indent));
            assert(~exist('level_at_indent','var') || tc.checkf(@tc.vector,level_at_indent));
            assert(~exist('level_at_indent','var') || tc.checkf(@(ii)length(level_at_indent{ii}) == (per_handler_indent(ii) / 2 + 1),1:length(level_at_indent)));
            assert(~exist('level_at_indent','var') || tc.checkf(@tc.cell,level_at_indent));
            assert(~exist('level_at_indent','var') || tc.checkf(@(c)tc.checkf(@tc.logging_level,c),level_at_indent));
            assert((exist('level_stack','var') && exist('per_handler_indent','var') && exist('level_at_indent','var')) || ...
                   (~exist('level_stack','var') && ~exist('per_handler_indent','var') && ~exist('level_at_indent','var')));

            if exist('level_stack','var')
                level_stack_t = level_stack;
                per_handler_indent_t = per_handler_indent;
                level_at_indent_t = level_at_indent;
            else
                level_stack_t = logging.level.TopLevel;
                per_handler_indent_t = zeros(length(handlers),1);
                level_at_indent_t = repmat({{logging.level.TopLevel}},length(handlers),1);
            end
            
            obj.handlers = utils.force_col(handlers);
            obj.handlers_count = length(handlers);
            obj.active = true;
            obj.level_stack = level_stack_t;
            obj.per_handler_indent = per_handler_indent_t;
            obj.level_at_indent = level_at_indent_t;
        end
        
        function [] = message(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
                 
            message_lines = textscan(sprintf(message_fmt,varargin{:}),'%s','Delimiter',sprintf('\n'),'Whitespace','');
            
            for ii = 1:obj.handlers_count
                if obj.level_stack(end) >= obj.handlers{ii}.min_level
                    for jj = 1:length(message_lines{1})
                        message_full = sprintf('%s%s\n',repmat(' ',1,obj.per_handler_indent(ii)),message_lines{1}{jj});
                        obj.handlers{ii}.send(message_full);
                    end
                end
            end
        end
        
        function [] = beg_node(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            message = sprintf(message_fmt,varargin{:});
            
            for ii = 1:obj.handlers_count
                if obj.level_stack(end) >= obj.handlers{ii}.min_level
                    message_full = sprintf('%s%s:\n',repmat(' ',1,obj.per_handler_indent(ii)),message);
                    obj.per_handler_indent(ii) = obj.per_handler_indent(ii) + 2;
                    obj.level_at_indent{ii} = cat(1,obj.level_at_indent{ii},{obj.level_stack(end)});
                    obj.handlers{ii}.send(message_full);
                end
            end
        end
        
        function [] = end_node(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            
            for ii = 1:obj.handlers_count
                if obj.level_stack(end) >= obj.handlers{ii}.min_level
                    assert(obj.level_stack(end) == obj.level_at_indent{ii}{end});
                    
                    obj.per_handler_indent(ii) = obj.per_handler_indent(ii) - 2;
                    obj.level_at_indent{ii} = obj.level_at_indent{ii}(1:end-1);
                end
            end
        end
        
        function [] = beg_experiment(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.TopLevel);
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
            
            obj.level_stack = [obj.level_stack;logging.level.Experiment];
            obj.beg_node(message_fmt,varargin{:});
        end
        
        function [] = end_experiment(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Experiment);

            obj.end_node();
            obj.level_stack = obj.level_stack(1:end-1);
        end
        
        function [] = beg_architecture(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.TopLevel) || ...
                   (obj.level_stack(end) == logging.level.Experiment));
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            obj.level_stack = [obj.level_stack;logging.level.Architecture];
            obj.beg_node(message_fmt,varargin{:});
        end
        
        function [] = end_architecture(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Architecture);
            
            obj.end_node();
            obj.level_stack = obj.level_stack(1:end-1);
        end
        
        function [] = beg_transform(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.TopLevel) || ...
                   (obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Architecture));
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            obj.level_stack = [obj.level_stack;logging.level.Transform];
            obj.beg_node(message_fmt,varargin{:});
        end
        
        function [] = end_transform(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Transform);
            
            obj.end_node();
            obj.level_stack = obj.level_stack(1:end-1);
        end
        
        function [] = beg_classifier(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.TopLevel) || ...
                   (obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Architecture) || ...
                   (obj.level_stack(end) == logging.level.Classifier));
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            obj.level_stack = [obj.level_stack;logging.level.Classifier];
            obj.beg_node(message_fmt,varargin{:});
        end
        
        function [] = end_classifier(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Classifier);
            
            obj.end_node();
            obj.level_stack = obj.level_stack(1:end-1);
        end
        
        function [] = beg_dataset_io(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.TopLevel) || ...
                   (obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Architecture));
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            obj.level_stack = [obj.level_stack;logging.level.Dataset_IO];
            obj.beg_node(message_fmt,varargin{:});
        end
        
        function [] = end_dataset_io(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Dataset_IO);
            
            obj.end_node();
            obj.level_stack = obj.level_stack(1:end-1);
        end
        
        function [] = beg_results(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.TopLevel) || ...
                   (obj.level_stack(end) == logging.level.Experiment));
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            obj.level_stack = [obj.level_stack;logging.level.Results];
            obj.beg_node(message_fmt,varargin{:});
        end
        
        function [] = end_results(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Results);
            
            obj.end_node();
            obj.level_stack = obj.level_stack(1:end-1);
        end
        
        function [] = beg_error(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
                 
            obj.level_stack = [obj.level_stack;logging.level.Error];
            obj.beg_node(message_fmt,varargin{:});
        end
        
        function [] = end_error(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Error);
            
            obj.end_node();
            obj.level_stack = obj.level_stack(1:end-1);
        end
        
        function [new_logger] = new_node(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
            
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_node(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_experiment(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.TopLevel);
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
            
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_experiment(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_architecture(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.TopLevel) || ...
                   (obj.level_stack(end) == logging.level.Experiment));
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_architecture(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_transform(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.TopLevel) || ...
                   (obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Architecture));
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_transform(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_classifier(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.TopLevel) || ...
                   (obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Architecture) || ...
                   (obj.level_stack(end) == logging.level.Classifier));
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_classifier(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_dataset_io(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.TopLevel) || ...
                   (obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Architecture));
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_dataset_io(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_results(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.TopLevel) || ...
                   (obj.level_stack(end) == logging.level.Experiment));
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
               
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_results(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_error(obj,message_fmt,varargin)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(obj.active);
            assert(tc.scalar(message_fmt));
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.cell(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.value,varargin));
                 
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_error(message_fmt,varargin{:});
        end
        
        function [] = close(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));
            assert(tc.same(obj.level_stack,logging.level.TopLevel));
            assert(tc.same(obj.per_handler_indent,zeros(length(obj.handlers),1)));
            assert(tc.same(obj.level_at_indent,repmat({{logging.level.TopLevel}},length(obj.handlers),1)));
            
            if obj.active
                obj.active = false;
            end
        end
        
        function [] = delete(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_logger(obj));

            if (length(obj.level_stack) == 1) && ...
                (tc.same(obj.level_stack,logging.level.TopLevel)) && ...
                (tc.same(obj.per_handler_indent,zeros(length(obj.handlers),1))) && ...
                (tc.same(obj.level_at_indent,repmat({{logging.level.TopLevel}},length(obj.handlers),1)))
                obj.close();
            else
                % It is not safe to close the object in this state. Rather
                % than assert in a destructor, we'll let it slide.
            end
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logging.logger".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With one handler.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            assert(tc.same(length(log.handlers),1));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.All));
            assert(tc.same(log.handlers{1}.logged_data,''));
            assert(tc.same(log.handlers_count,1));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,0));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel}}));
                        
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With three handlers.\n');
            
            hnd1 = logging.handlers.testing(logging.level.All);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Dataset_IO);
            log = logging.logger({hnd1,hnd2,hnd3});
            
            assert(tc.same(length(log.handlers),3));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.All));
            assert(tc.same(log.handlers{1}.logged_data,''));
            assert(tc.same(log.handlers{2}.active,true));
            assert(tc.same(log.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log.handlers{2}.logged_data,''));
            assert(tc.same(log.handlers{3}.active,true));
            assert(tc.same(log.handlers{3}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log.handlers{3}.logged_data,''));
            assert(tc.same(log.handlers_count,3));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};{logging.level.TopLevel};{logging.level.TopLevel}}));
            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            
            clearvars -except display;
            
            fprintf('    With specified initial state.\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Dataset_IO);
            log = logging.logger({hnd1,hnd2,hnd3},[logging.level.TopLevel;logging.level.Experiment],[2;4;4],...
                                                  {{logging.level.TopLevel;logging.level.TopLevel};...
                                                   {logging.level.TopLevel;logging.level.TopLevel;logging.level.Experiment};...
                                                   {logging.level.TopLevel;logging.level.TopLevel;logging.level.Experiment}});
                                               
            assert(tc.same(length(log.handlers),3));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log.handlers{1}.logged_data,''));
            assert(tc.same(log.handlers{2}.active,true));
            assert(tc.same(log.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log.handlers{2}.logged_data,''));
            assert(tc.same(log.handlers{3}.active,true));
            assert(tc.same(log.handlers{3}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log.handlers{3}.logged_data,''));
            assert(tc.same(log.handlers_count,3));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,[logging.level.TopLevel;logging.level.Experiment]));
            assert(tc.same(log.per_handler_indent,[2;4;4]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel;logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.TopLevel;logging.level.Experiment};...
                                                {logging.level.TopLevel;logging.level.TopLevel;logging.level.Experiment}}));

            log.end_experiment();
            log.end_node();
            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            
            clearvars -except display;
            
            fprintf('  Function "message".\n');
            
            fprintf('    Singleline messages and different levels.\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log.message('A');
            log.beg_experiment('1');
            log.message('1A');
            log.end_experiment();
            log.beg_architecture('2');
            log.message('2A');
            log.end_architecture();
            log.beg_transform('3');
            log.message('3A');
            log.end_transform();
            log.beg_classifier('4');
            log.message('4A');
            log.end_classifier();
            log.beg_dataset_io('5');
            log.message('5A');
            log.end_dataset_io();
            log.beg_results('6');
            log.message('6A');
            log.end_results();
            log.beg_error('7');
            log.message('7A');
            log.end_error();
            log.message('B');
            
            assert(tc.same(hnd1.logged_data,sprintf('A\n6:\n  6A\n7:\n  7A\nB\n')));
            assert(tc.same(hnd2.logged_data,sprintf('A\n1:\n  1A\n6:\n  6A\n7:\n  7A\nB\n')));
            assert(tc.same(hnd3.logged_data,sprintf('A\n1:\n  1A\n2:\n  2A\n6:\n  6A\n7:\n  7A\nB\n')));
            assert(tc.same(hnd4.logged_data,sprintf('A\n1:\n  1A\n2:\n  2A\n3:\n  3A\n6:\n  6A\n7:\n  7A\nB\n')));
            assert(tc.same(hnd5.logged_data,sprintf('A\n1:\n  1A\n2:\n  2A\n4:\n  4A\n6:\n  6A\n7:\n  7A\nB\n')));
            assert(tc.same(hnd6.logged_data,sprintf('A\n1:\n  1A\n2:\n  2A\n5:\n  5A\n6:\n  6A\n7:\n  7A\nB\n')));
            assert(tc.same(hnd7.logged_data,sprintf('6:\n  6A\n')));
            assert(tc.same(hnd8.logged_data,sprintf('7:\n  7A\n')));
            assert(tc.same(hnd9.logged_data,sprintf('A\n1:\n  1A\n2:\n  2A\n3:\n  3A\n4:\n  4A\n5:\n  5A\n6:\n  6A\n7:\n  7A\nB\n')));
            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('    Multiline messages.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            log.message(sprintf('Hello\nWorld'));
            log.beg_node('A');
            log.message(sprintf('Hello\nWorld'));
            log.end_node();
            
            assert(tc.same(hnd.logged_data,sprintf('Hello\nWorld\nA:\n  Hello\n  World\n')));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Functions "beg_node" and "end_node".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log.beg_node('A');
            
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[2;2;2;2;2;2;0;0;2]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel;logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.TopLevel}}));
                                            
            log.end_node();
            
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();

            clearvars -except display;
            
            fprintf('  Functions "beg_experiment" and "end_experiment".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log.beg_experiment('A');
            
            assert(tc.same(log.level_stack,[logging.level.TopLevel;logging.level.Experiment]));
            assert(tc.same(log.per_handler_indent,[0;2;2;2;2;2;0;0;2]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Experiment};...
                                                {logging.level.TopLevel;logging.level.Experiment};...
                                                {logging.level.TopLevel;logging.level.Experiment};...
                                                {logging.level.TopLevel;logging.level.Experiment};...
                                                {logging.level.TopLevel;logging.level.Experiment};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Experiment}}));
                                            
            log.end_experiment();
            
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();

            clearvars -except display;
            
            fprintf('  Functions "beg_architecture" and "end_architecture".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log.beg_architecture('A');
            
            assert(tc.same(log.level_stack,[logging.level.TopLevel;logging.level.Architecture]));
            assert(tc.same(log.per_handler_indent,[0;0;2;2;2;2;0;0;2]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Architecture};...
                                                {logging.level.TopLevel;logging.level.Architecture};...
                                                {logging.level.TopLevel;logging.level.Architecture};...
                                                {logging.level.TopLevel;logging.level.Architecture};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Architecture}}));
                                            
            log.end_architecture();
            
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Functions "beg_transform" and "end_transform".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log.beg_transform('A');
            
            assert(tc.same(log.level_stack,[logging.level.TopLevel;logging.level.Transform]));
            assert(tc.same(log.per_handler_indent,[0;0;0;2;0;0;0;0;2]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Transform};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Transform}}));
                                            
            log.end_transform();
            
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Functions "beg_classifier" and "end_classifier".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log.beg_classifier('A');
            
            assert(tc.same(log.level_stack,[logging.level.TopLevel;logging.level.Classifier]));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;2;0;0;0;2]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Classifier};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Classifier}}));
                                            
            log.end_classifier();
            
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Functions "beg_dataset_io" and "end_dataset_io".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log.beg_dataset_io('A');
            
            assert(tc.same(log.level_stack,[logging.level.TopLevel;logging.level.Dataset_IO]));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;2;0;0;2]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Dataset_IO};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Dataset_IO}}));
                                            
            log.end_dataset_io();
            
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Functions "beg_results" and "end_results".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log.beg_results('A');
            
            assert(tc.same(log.level_stack,[logging.level.TopLevel;logging.level.Results]));
            assert(tc.same(log.per_handler_indent,[2;2;2;2;2;2;2;0;2]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel;logging.level.Results};...
                                                {logging.level.TopLevel;logging.level.Results};...
                                                {logging.level.TopLevel;logging.level.Results};...
                                                {logging.level.TopLevel;logging.level.Results};...
                                                {logging.level.TopLevel;logging.level.Results};...
                                                {logging.level.TopLevel;logging.level.Results};...
                                                {logging.level.TopLevel;logging.level.Results};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Results}}));
                                            
            log.end_results();
            
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Functions "beg_error" and "end_error".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log.beg_error('A');
            
            assert(tc.same(log.level_stack,[logging.level.TopLevel;logging.level.Error]));
            assert(tc.same(log.per_handler_indent,[2;2;2;2;2;2;0;2;2]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel;logging.level.Error};...
                                                {logging.level.TopLevel;logging.level.Error};...
                                                {logging.level.TopLevel;logging.level.Error};...
                                                {logging.level.TopLevel;logging.level.Error};...
                                                {logging.level.TopLevel;logging.level.Error};...
                                                {logging.level.TopLevel;logging.level.Error};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel;logging.level.Error};...
                                                {logging.level.TopLevel;logging.level.Error}}));
                                            
            log.end_error();
            
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Function "new_node".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log_a = log.new_node('A');
            
            assert(tc.same(length(log_a.handlers),9));
            assert(tc.same(log_a.handlers{1}.active,true));
            assert(tc.same(log_a.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log_a.handlers{1}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{2}.active,true));
            assert(tc.same(log_a.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log_a.handlers{2}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{3}.active,true));
            assert(tc.same(log_a.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log_a.handlers{3}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{4}.active,true));
            assert(tc.same(log_a.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{5}.active,true));
            assert(tc.same(log_a.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{6}.active,true));
            assert(tc.same(log_a.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log_a.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{7}.active,true));
            assert(tc.same(log_a.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log_a.handlers{7}.logged_data,''));
            assert(tc.same(log_a.handlers{8}.active,true));
            assert(tc.same(log_a.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log_a.handlers{8}.logged_data,''));
            assert(tc.same(log_a.handlers{9}.active,true));
            assert(tc.same(log_a.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log_a.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers_count,9));
            assert(tc.same(log_a.active,true));
            assert(tc.same(log_a.level_stack,[logging.level.TopLevel]));
            assert(tc.same(log_a.per_handler_indent,[2;2;2;2;2;2;0;0;2]));
            assert(tc.same(log_a.level_at_indent,{{logging.level.TopLevel;logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.TopLevel}}));
            assert(tc.same(length(log.handlers),9));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log.handlers{1}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{2}.active,true));
            assert(tc.same(log.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log.handlers{2}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{3}.active,true));
            assert(tc.same(log.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log.handlers{3}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{4}.active,true));
            assert(tc.same(log.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{5}.active,true));
            assert(tc.same(log.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{6}.active,true));
            assert(tc.same(log.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{7}.active,true));
            assert(tc.same(log.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log.handlers{7}.logged_data,''));
            assert(tc.same(log.handlers{8}.active,true));
            assert(tc.same(log.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log.handlers{8}.logged_data,''));
            assert(tc.same(log.handlers{9}.active,true));
            assert(tc.same(log.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers_count,9));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();

            clearvars -except display;
            
            fprintf('  Function "new_experiment".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log_a = log.new_experiment('A');
            
            assert(tc.same(length(log_a.handlers),9));
            assert(tc.same(log_a.handlers{1}.active,true));
            assert(tc.same(log_a.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log_a.handlers{1}.logged_data,''));
            assert(tc.same(log_a.handlers{2}.active,true));
            assert(tc.same(log_a.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log_a.handlers{2}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{3}.active,true));
            assert(tc.same(log_a.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log_a.handlers{3}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{4}.active,true));
            assert(tc.same(log_a.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{5}.active,true));
            assert(tc.same(log_a.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{6}.active,true));
            assert(tc.same(log_a.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log_a.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{7}.active,true));
            assert(tc.same(log_a.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log_a.handlers{7}.logged_data,''));
            assert(tc.same(log_a.handlers{8}.active,true));
            assert(tc.same(log_a.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log_a.handlers{8}.logged_data,''));
            assert(tc.same(log_a.handlers{9}.active,true));
            assert(tc.same(log_a.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log_a.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers_count,9));
            assert(tc.same(log_a.active,true));
            assert(tc.same(log_a.level_stack,[logging.level.TopLevel;logging.level.Experiment]));
            assert(tc.same(log_a.per_handler_indent,[0;2;2;2;2;2;0;0;2]));
            assert(tc.same(log_a.level_at_indent,{{logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Experiment};...
                                                  {logging.level.TopLevel;logging.level.Experiment};...
                                                  {logging.level.TopLevel;logging.level.Experiment};...
                                                  {logging.level.TopLevel;logging.level.Experiment};...
                                                  {logging.level.TopLevel;logging.level.Experiment};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Experiment}}));
            assert(tc.same(length(log.handlers),9));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log.handlers{1}.logged_data,''));
            assert(tc.same(log.handlers{2}.active,true));
            assert(tc.same(log.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log.handlers{2}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{3}.active,true));
            assert(tc.same(log.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log.handlers{3}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{4}.active,true));
            assert(tc.same(log.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{5}.active,true));
            assert(tc.same(log.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{6}.active,true));
            assert(tc.same(log.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{7}.active,true));
            assert(tc.same(log.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log.handlers{7}.logged_data,''));
            assert(tc.same(log.handlers{8}.active,true));
            assert(tc.same(log.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log.handlers{8}.logged_data,''));
            assert(tc.same(log.handlers{9}.active,true));
            assert(tc.same(log.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers_count,9));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();

            clearvars -except display;
            
            fprintf('  Function "new_architecture".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log_a = log.new_architecture('A');
            
            assert(tc.same(length(log_a.handlers),9));
            assert(tc.same(log_a.handlers{1}.active,true));
            assert(tc.same(log_a.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log_a.handlers{1}.logged_data,''));
            assert(tc.same(log_a.handlers{2}.active,true));
            assert(tc.same(log_a.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log_a.handlers{2}.logged_data,''));
            assert(tc.same(log_a.handlers{3}.active,true));
            assert(tc.same(log_a.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log_a.handlers{3}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{4}.active,true));
            assert(tc.same(log_a.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{5}.active,true));
            assert(tc.same(log_a.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{6}.active,true));
            assert(tc.same(log_a.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log_a.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{7}.active,true));
            assert(tc.same(log_a.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log_a.handlers{7}.logged_data,''));
            assert(tc.same(log_a.handlers{8}.active,true));
            assert(tc.same(log_a.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log_a.handlers{8}.logged_data,''));
            assert(tc.same(log_a.handlers{9}.active,true));
            assert(tc.same(log_a.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log_a.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers_count,9));
            assert(tc.same(log_a.active,true));
            assert(tc.same(log_a.level_stack,[logging.level.TopLevel;logging.level.Architecture]));
            assert(tc.same(log_a.per_handler_indent,[0;0;2;2;2;2;0;0;2]));
            assert(tc.same(log_a.level_at_indent,{{logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Architecture};...
                                                  {logging.level.TopLevel;logging.level.Architecture};...
                                                  {logging.level.TopLevel;logging.level.Architecture};...
                                                  {logging.level.TopLevel;logging.level.Architecture};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Architecture}}));
            assert(tc.same(length(log.handlers),9));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log.handlers{1}.logged_data,''));
            assert(tc.same(log.handlers{2}.active,true));
            assert(tc.same(log.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log.handlers{2}.logged_data,''));
            assert(tc.same(log.handlers{3}.active,true));
            assert(tc.same(log.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log.handlers{3}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{4}.active,true));
            assert(tc.same(log.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{5}.active,true));
            assert(tc.same(log.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{6}.active,true));
            assert(tc.same(log.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{7}.active,true));
            assert(tc.same(log.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log.handlers{7}.logged_data,''));
            assert(tc.same(log.handlers{8}.active,true));
            assert(tc.same(log.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log.handlers{8}.logged_data,''));
            assert(tc.same(log.handlers{9}.active,true));
            assert(tc.same(log.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers_count,9));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Function "new_transform".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log_a = log.new_transform('A');
            
            assert(tc.same(length(log_a.handlers),9));
            assert(tc.same(log_a.handlers{1}.active,true));
            assert(tc.same(log_a.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log_a.handlers{1}.logged_data,''));
            assert(tc.same(log_a.handlers{2}.active,true));
            assert(tc.same(log_a.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log_a.handlers{2}.logged_data,''));
            assert(tc.same(log_a.handlers{3}.active,true));
            assert(tc.same(log_a.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log_a.handlers{3}.logged_data,''));
            assert(tc.same(log_a.handlers{4}.active,true));
            assert(tc.same(log_a.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{5}.active,true));
            assert(tc.same(log_a.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log_a.handlers{5}.logged_data,''));
            assert(tc.same(log_a.handlers{6}.active,true));
            assert(tc.same(log_a.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log_a.handlers{6}.logged_data,''));
            assert(tc.same(log_a.handlers{7}.active,true));
            assert(tc.same(log_a.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log_a.handlers{7}.logged_data,''));
            assert(tc.same(log_a.handlers{8}.active,true));
            assert(tc.same(log_a.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log_a.handlers{8}.logged_data,''));
            assert(tc.same(log_a.handlers{9}.active,true));
            assert(tc.same(log_a.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log_a.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers_count,9));
            assert(tc.same(log_a.active,true));
            assert(tc.same(log_a.level_stack,[logging.level.TopLevel;logging.level.Transform]));
            assert(tc.same(log_a.per_handler_indent,[0;0;0;2;0;0;0;0;2]));
            assert(tc.same(log_a.level_at_indent,{{logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Transform};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Transform}}));
            assert(tc.same(length(log.handlers),9));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log.handlers{1}.logged_data,''));
            assert(tc.same(log.handlers{2}.active,true));
            assert(tc.same(log.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log.handlers{2}.logged_data,''));
            assert(tc.same(log.handlers{3}.active,true));
            assert(tc.same(log.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log.handlers{3}.logged_data,''));
            assert(tc.same(log.handlers{4}.active,true));
            assert(tc.same(log.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{5}.active,true));
            assert(tc.same(log.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log.handlers{5}.logged_data,''));
            assert(tc.same(log.handlers{6}.active,true));
            assert(tc.same(log.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log.handlers{6}.logged_data,''));
            assert(tc.same(log.handlers{7}.active,true));
            assert(tc.same(log.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log.handlers{7}.logged_data,''));
            assert(tc.same(log.handlers{8}.active,true));
            assert(tc.same(log.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log.handlers{8}.logged_data,''));
            assert(tc.same(log.handlers{9}.active,true));
            assert(tc.same(log.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers_count,9));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Function "new_classifier".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log_a = log.new_classifier('A');
            
            assert(tc.same(length(log_a.handlers),9));
            assert(tc.same(log_a.handlers{1}.active,true));
            assert(tc.same(log_a.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log_a.handlers{1}.logged_data,''));
            assert(tc.same(log_a.handlers{2}.active,true));
            assert(tc.same(log_a.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log_a.handlers{2}.logged_data,''));
            assert(tc.same(log_a.handlers{3}.active,true));
            assert(tc.same(log_a.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log_a.handlers{3}.logged_data,''));
            assert(tc.same(log_a.handlers{4}.active,true));
            assert(tc.same(log_a.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log_a.handlers{4}.logged_data,''));
            assert(tc.same(log_a.handlers{5}.active,true));
            assert(tc.same(log_a.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{6}.active,true));
            assert(tc.same(log_a.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log_a.handlers{6}.logged_data,''));
            assert(tc.same(log_a.handlers{7}.active,true));
            assert(tc.same(log_a.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log_a.handlers{7}.logged_data,''));
            assert(tc.same(log_a.handlers{8}.active,true));
            assert(tc.same(log_a.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log_a.handlers{8}.logged_data,''));
            assert(tc.same(log_a.handlers{9}.active,true));
            assert(tc.same(log_a.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log_a.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers_count,9));
            assert(tc.same(log_a.active,true));
            assert(tc.same(log_a.level_stack,[logging.level.TopLevel;logging.level.Classifier]));
            assert(tc.same(log_a.per_handler_indent,[0;0;0;0;2;0;0;0;2]));
            assert(tc.same(log_a.level_at_indent,{{logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Classifier};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Classifier}}));
            assert(tc.same(length(log.handlers),9));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log.handlers{1}.logged_data,''));
            assert(tc.same(log.handlers{2}.active,true));
            assert(tc.same(log.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log.handlers{2}.logged_data,''));
            assert(tc.same(log.handlers{3}.active,true));
            assert(tc.same(log.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log.handlers{3}.logged_data,''));
            assert(tc.same(log.handlers{4}.active,true));
            assert(tc.same(log.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log.handlers{4}.logged_data,''));
            assert(tc.same(log.handlers{5}.active,true));
            assert(tc.same(log.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{6}.active,true));
            assert(tc.same(log.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log.handlers{6}.logged_data,''));
            assert(tc.same(log.handlers{7}.active,true));
            assert(tc.same(log.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log.handlers{7}.logged_data,''));
            assert(tc.same(log.handlers{8}.active,true));
            assert(tc.same(log.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log.handlers{8}.logged_data,''));
            assert(tc.same(log.handlers{9}.active,true));
            assert(tc.same(log.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers_count,9));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Function "new_dataset_io".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log_a = log.new_dataset_io('A');
            
            assert(tc.same(length(log_a.handlers),9));
            assert(tc.same(log_a.handlers{1}.active,true));
            assert(tc.same(log_a.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log_a.handlers{1}.logged_data,''));
            assert(tc.same(log_a.handlers{2}.active,true));
            assert(tc.same(log_a.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log_a.handlers{2}.logged_data,''));
            assert(tc.same(log_a.handlers{3}.active,true));
            assert(tc.same(log_a.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log_a.handlers{3}.logged_data,''));
            assert(tc.same(log_a.handlers{4}.active,true));
            assert(tc.same(log_a.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log_a.handlers{4}.logged_data,''));
            assert(tc.same(log_a.handlers{5}.active,true));
            assert(tc.same(log_a.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log_a.handlers{5}.logged_data,''));
            assert(tc.same(log_a.handlers{6}.active,true));
            assert(tc.same(log_a.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log_a.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{7}.active,true));
            assert(tc.same(log_a.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log_a.handlers{7}.logged_data,''));
            assert(tc.same(log_a.handlers{8}.active,true));
            assert(tc.same(log_a.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log_a.handlers{8}.logged_data,''));
            assert(tc.same(log_a.handlers{9}.active,true));
            assert(tc.same(log_a.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log_a.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers_count,9));
            assert(tc.same(log_a.active,true));
            assert(tc.same(log_a.level_stack,[logging.level.TopLevel;logging.level.Dataset_IO]));
            assert(tc.same(log_a.per_handler_indent,[0;0;0;0;0;2;0;0;2]));
            assert(tc.same(log_a.level_at_indent,{{logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Dataset_IO};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Dataset_IO}}));
            assert(tc.same(length(log.handlers),9));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log.handlers{1}.logged_data,''));
            assert(tc.same(log.handlers{2}.active,true));
            assert(tc.same(log.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log.handlers{2}.logged_data,''));
            assert(tc.same(log.handlers{3}.active,true));
            assert(tc.same(log.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log.handlers{3}.logged_data,''));
            assert(tc.same(log.handlers{4}.active,true));
            assert(tc.same(log.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log.handlers{4}.logged_data,''));
            assert(tc.same(log.handlers{5}.active,true));
            assert(tc.same(log.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log.handlers{5}.logged_data,''));
            assert(tc.same(log.handlers{6}.active,true));
            assert(tc.same(log.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{7}.active,true));
            assert(tc.same(log.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log.handlers{7}.logged_data,''));
            assert(tc.same(log.handlers{8}.active,true));
            assert(tc.same(log.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log.handlers{8}.logged_data,''));
            assert(tc.same(log.handlers{9}.active,true));
            assert(tc.same(log.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers_count,9));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Function "new_results".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log_a = log.new_results('A');
            
            assert(tc.same(length(log_a.handlers),9));
            assert(tc.same(log_a.handlers{1}.active,true));
            assert(tc.same(log_a.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log_a.handlers{1}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{2}.active,true));
            assert(tc.same(log_a.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log_a.handlers{2}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{3}.active,true));
            assert(tc.same(log_a.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log_a.handlers{3}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{4}.active,true));
            assert(tc.same(log_a.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{5}.active,true));
            assert(tc.same(log_a.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{6}.active,true));
            assert(tc.same(log_a.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log_a.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{7}.active,true));
            assert(tc.same(log_a.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log_a.handlers{7}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{8}.active,true));
            assert(tc.same(log_a.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log_a.handlers{8}.logged_data,''));
            assert(tc.same(log_a.handlers{9}.active,true));
            assert(tc.same(log_a.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log_a.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers_count,9));
            assert(tc.same(log_a.active,true));
            assert(tc.same(log_a.level_stack,[logging.level.TopLevel;logging.level.Results]));
            assert(tc.same(log_a.per_handler_indent,[2;2;2;2;2;2;2;0;2]));
            assert(tc.same(log_a.level_at_indent,{{logging.level.TopLevel;logging.level.Results};...
                                                  {logging.level.TopLevel;logging.level.Results};...
                                                  {logging.level.TopLevel;logging.level.Results};...
                                                  {logging.level.TopLevel;logging.level.Results};...
                                                  {logging.level.TopLevel;logging.level.Results};...
                                                  {logging.level.TopLevel;logging.level.Results};...
                                                  {logging.level.TopLevel;logging.level.Results};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Results}}));
            assert(tc.same(length(log.handlers),9));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log.handlers{1}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{2}.active,true));
            assert(tc.same(log.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log.handlers{2}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{3}.active,true));
            assert(tc.same(log.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log.handlers{3}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{4}.active,true));
            assert(tc.same(log.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{5}.active,true));
            assert(tc.same(log.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{6}.active,true));
            assert(tc.same(log.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{7}.active,true));
            assert(tc.same(log.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log.handlers{7}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{8}.active,true));
            assert(tc.same(log.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log.handlers{8}.logged_data,''));
            assert(tc.same(log.handlers{9}.active,true));
            assert(tc.same(log.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers_count,9));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Function "new_error".\n');
            
            hnd1 = logging.handlers.testing(logging.level.TopLevel);
            hnd2 = logging.handlers.testing(logging.level.Experiment);
            hnd3 = logging.handlers.testing(logging.level.Architecture);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Classifier);
            hnd6 = logging.handlers.testing(logging.level.Dataset_IO);
            hnd7 = logging.handlers.testing(logging.level.Results);
            hnd8 = logging.handlers.testing(logging.level.Error);
            hnd9 = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5,hnd6,hnd7,hnd8,hnd9});
            
            log_a = log.new_error('A');
            
            assert(tc.same(length(log_a.handlers),9));
            assert(tc.same(log_a.handlers{1}.active,true));
            assert(tc.same(log_a.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log_a.handlers{1}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{2}.active,true));
            assert(tc.same(log_a.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log_a.handlers{2}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{3}.active,true));
            assert(tc.same(log_a.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log_a.handlers{3}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{4}.active,true));
            assert(tc.same(log_a.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{5}.active,true));
            assert(tc.same(log_a.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{6}.active,true));
            assert(tc.same(log_a.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log_a.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{7}.active,true));
            assert(tc.same(log_a.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log_a.handlers{7}.logged_data,''));
            assert(tc.same(log_a.handlers{8}.active,true));
            assert(tc.same(log_a.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log_a.handlers{8}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers{9}.active,true));
            assert(tc.same(log_a.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log_a.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log_a.handlers_count,9));
            assert(tc.same(log_a.active,true));
            assert(tc.same(log_a.level_stack,[logging.level.TopLevel;logging.level.Error]));
            assert(tc.same(log_a.per_handler_indent,[2;2;2;2;2;2;0;2;2]));
            assert(tc.same(log_a.level_at_indent,{{logging.level.TopLevel;logging.level.Error};...
                                                  {logging.level.TopLevel;logging.level.Error};...
                                                  {logging.level.TopLevel;logging.level.Error};...
                                                  {logging.level.TopLevel;logging.level.Error};...
                                                  {logging.level.TopLevel;logging.level.Error};...
                                                  {logging.level.TopLevel;logging.level.Error};...
                                                  {logging.level.TopLevel};...
                                                  {logging.level.TopLevel;logging.level.Error};...
                                                  {logging.level.TopLevel;logging.level.Error}}));
            assert(tc.same(length(log.handlers),9));
            assert(tc.same(log.handlers{1}.active,true));
            assert(tc.same(log.handlers{1}.min_level,logging.level.TopLevel));
            assert(tc.same(log.handlers{1}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{2}.active,true));
            assert(tc.same(log.handlers{2}.min_level,logging.level.Experiment));
            assert(tc.same(log.handlers{2}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{3}.active,true));
            assert(tc.same(log.handlers{3}.min_level,logging.level.Architecture));
            assert(tc.same(log.handlers{3}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{4}.active,true));
            assert(tc.same(log.handlers{4}.min_level,logging.level.Transform));
            assert(tc.same(log.handlers{4}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{5}.active,true));
            assert(tc.same(log.handlers{5}.min_level,logging.level.Classifier));
            assert(tc.same(log.handlers{5}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{6}.active,true));
            assert(tc.same(log.handlers{6}.min_level,logging.level.Dataset_IO));
            assert(tc.same(log.handlers{6}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{7}.active,true));
            assert(tc.same(log.handlers{7}.min_level,logging.level.Results));
            assert(tc.same(log.handlers{7}.logged_data,''));
            assert(tc.same(log.handlers{8}.active,true));
            assert(tc.same(log.handlers{8}.min_level,logging.level.Error));
            assert(tc.same(log.handlers{8}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers{9}.active,true));
            assert(tc.same(log.handlers{9}.min_level,logging.level.All));
            assert(tc.same(log.handlers{9}.logged_data,sprintf('A:\n')));
            assert(tc.same(log.handlers_count,9));
            assert(tc.same(log.active,true));
            assert(tc.same(log.level_stack,logging.level.TopLevel));
            assert(tc.same(log.per_handler_indent,[0;0;0;0;0;0;0;0;0]));
            assert(tc.same(log.level_at_indent,{{logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel};...
                                                {logging.level.TopLevel}}));
                                            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            hnd6.close();
            hnd7.close();
            hnd8.close();
            hnd9.close();
            
            clearvars -except display;
            
            fprintf('  Function "close".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
                        
            log.close();
            
            assert(tc.same(log.active,false));
            
            log.close();
            
            assert(tc.same(log.active,false));
            
            hnd.close();
            
            clearvars -except display;
        end
    end
end
