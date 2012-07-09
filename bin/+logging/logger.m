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
            assert(check.vector(handlers));
            assert(check.cell(handlers));
            assert(check.checkf(@check.scalar,handlers));
            assert(check.checkf(@check.logging_handler,handlers));
            assert(check.checkf(@(h)h.active,handlers));
            assert(~exist('level_stack','var') || check.vector(level_stack));
            assert(~exist('level_stack','var') || check.logging_level(level_stack));
            assert(~exist('per_handler_indent','var') || check.vector(per_handler_indent));
            assert(~exist('per_handler_indent','var') || (length(per_handler_indent) == length(handlers)));
            assert(~exist('per_handler_indent','var') || check.natural(per_handler_indent));
            assert(~exist('level_at_indent','var') || check.vector(level_at_indent));
            assert(~exist('level_at_indent','var') || (length(level_at_indent) == length(handlers)));
            assert(~exist('level_at_indent','var') || check.cell(level_at_indent));
            assert(~exist('level_at_indent','var') || check.checkf(@check.vector,level_at_indent));
            assert(~exist('level_at_indent','var') || check.checkf(@(ii)length(level_at_indent{ii}) == (per_handler_indent(ii) / 2 + 1),1:length(level_at_indent)));
            assert(~exist('level_at_indent','var') || check.checkf(@check.cell,level_at_indent));
            assert(~exist('level_at_indent','var') || check.checkf(@(c)check.checkf(@check.logging_level,c),level_at_indent));
            assert((exist('level_stack','var') && exist('per_handler_indent','var') && exist('level_at_indent','var')) || ...
                   (~exist('level_stack','var') && ~exist('per_handler_indent','var') && ~exist('level_at_indent','var')));

            if exist('level_stack','var')
                level_stack_t = level_stack;
                per_handler_indent_t = per_handler_indent;
                level_at_indent_t = level_at_indent;
            else
                level_stack_t = logging.level.Experiment;
                per_handler_indent_t = zeros(length(handlers),1);
                level_at_indent_t = repmat({{logging.level.Experiment}},length(handlers),1);
            end
            
            obj.handlers = utils.common.force_col(handlers);
            obj.handlers_count = length(handlers);
            obj.active = true;
            obj.level_stack = level_stack_t;
            obj.per_handler_indent = per_handler_indent_t;
            obj.level_at_indent = level_at_indent_t;
        end
        
        function [] = message(obj,message_fmt,varargin)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));
                 
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
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));
               
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
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            
            for ii = 1:obj.handlers_count
                if obj.level_stack(end) >= obj.handlers{ii}.min_level
                    assert(obj.level_stack(end) == obj.level_at_indent{ii}{end});
                    
                    obj.per_handler_indent(ii) = obj.per_handler_indent(ii) - 2;
                    obj.level_at_indent{ii} = obj.level_at_indent{ii}(1:end-1);
                end
            end
        end

        function [] = beg_classifier(obj,message_fmt,varargin)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Classifier));
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));
               
            obj.beg_node(message_fmt,varargin{:});
            obj.level_stack = [obj.level_stack;logging.level.Classifier];
        end
        
        function [] = end_classifier(obj)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Classifier);
            
            obj.level_stack = obj.level_stack(1:end-1);
            obj.end_node();
        end
        
        function [] = beg_regressor(obj,message_fmt,varargin)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Regressor));
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));
               
            obj.beg_node(message_fmt,varargin{:});
            obj.level_stack = [obj.level_stack;logging.level.Regressor];
        end
        
        function [] = end_regressor(obj)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Regressor);
            
            obj.level_stack = obj.level_stack(1:end-1);
            obj.end_node();
        end
        
        function [] = beg_transform(obj,message_fmt,varargin)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Transform));
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));

            obj.beg_node(message_fmt,varargin{:});
            obj.level_stack = [obj.level_stack;logging.level.Transform];
        end
        
        function [] = end_transform(obj)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Transform);
            
            obj.level_stack = obj.level_stack(1:end-1);
            obj.end_node();
        end
        
        function [] = beg_results(obj,message_fmt,varargin)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Experiment);
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));
               
            obj.beg_node(message_fmt,varargin{:});
            obj.level_stack = [obj.level_stack;logging.level.Results];
        end
        
        function [] = end_results(obj)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Results);
            
            obj.level_stack = obj.level_stack(1:end-1);
            obj.end_node();
        end
        
        function [new_logger] = new_node(obj,message_fmt,varargin)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));
            
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_node(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_classifier(obj,message_fmt,varargin)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Classifier));
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));
               
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_classifier(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_regressor(obj,message_fmt,varargin)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Regressor));
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));
               
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_regressor(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_transform(obj,message_fmt,varargin)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert((obj.level_stack(end) == logging.level.Experiment) || ...
                   (obj.level_stack(end) == logging.level.Transform));
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));
               
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_transform(message_fmt,varargin{:});
        end
        
        function [new_logger] = new_results(obj,message_fmt,varargin)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(obj.active);
            assert(obj.level_stack(end) == logging.level.Experiment);
            assert(check.scalar(message_fmt));
            assert(check.string(message_fmt));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.cell(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.value,varargin));
               
            new_logger = logging.logger(obj.handlers,obj.level_stack,obj.per_handler_indent,obj.level_at_indent);
            new_logger.beg_results(message_fmt,varargin{:});
        end
        
        function [] = close(obj)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));
            assert(check.same(obj.level_stack,logging.level.Experiment));
            assert(check.same(obj.per_handler_indent,zeros(length(obj.handlers),1)));
            assert(check.same(obj.level_at_indent,repmat({{logging.level.Experiment}},length(obj.handlers),1)));
            
            if obj.active
                obj.active = false;
            end
        end
        
        function [] = delete(obj)
            assert(check.scalar(obj));
            assert(check.logging_logger(obj));

            if (length(obj.level_stack) == 1) && ...
                (check.same(obj.level_stack,logging.level.Experiment)) && ...
                (check.same(obj.per_handler_indent,zeros(length(obj.handlers),1))) && ...
                (check.same(obj.level_at_indent,repmat({{logging.level.Experiment}},length(obj.handlers),1)))
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
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg =  logging.logger({hnd});
            
            assert(check.same(length(logg.handlers),1));
            assert(check.same(logg.handlers{1}.active,true));
            assert(check.same(logg.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg.handlers{1}.logged_data,''));
            assert(check.same(logg.handlers_count,1));
            assert(check.same(logg.active,true));
            assert(check.same(logg.level_stack,logging.level.Experiment));
            assert(check.same(logg.per_handler_indent,0));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment}}));
                        
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With three handlers.\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3});
            
            assert(check.same(length(logg.handlers),3));
            assert(check.same(logg.handlers{1}.active,true));
            assert(check.same(logg.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg.handlers{1}.logged_data,''));
            assert(check.same(logg.handlers{2}.active,true));
            assert(check.same(logg.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg.handlers{2}.logged_data,''));
            assert(check.same(logg.handlers{3}.active,true));
            assert(check.same(logg.handlers{3}.min_level,logging.level.Results));
            assert(check.same(logg.handlers{3}.logged_data,''));
            assert(check.same(logg.handlers_count,3));
            assert(check.same(logg.active,true));
            assert(check.same(logg.level_stack,logging.level.Experiment));
            assert(check.same(logg.per_handler_indent,[0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};{logging.level.Experiment};{logging.level.Experiment}}));
            
            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "message".\n');
            
            fprintf('    Singleline messages and different levels.\n');

            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg.message('A');
            logg.beg_classifier('1');
            logg.message('B');
            logg.end_classifier();
            logg.beg_regressor('2');
            logg.message('C');
            logg.end_regressor();
            logg.beg_transform('3');
            logg.message('D');
            logg.end_transform();
            logg.beg_results('4');
            logg.message('E');
            logg.end_results();
            logg.message('F');
            
            assert(check.same(hnd1.logged_data,sprintf('A\n1:\n2:\n3:\n4:\n  E\nF\n')));
            assert(check.same(hnd2.logged_data,sprintf('A\n1:\n  B\n2:\n3:\n4:\n  E\nF\n')));
            assert(check.same(hnd3.logged_data,sprintf('A\n1:\n2:\n  C\n3:\n4:\n  E\nF\n')));
            assert(check.same(hnd4.logged_data,sprintf('A\n1:\n2:\n3:\n  D\n4:\n  E\nF\n')));
            assert(check.same(hnd5.logged_data,sprintf('A\n1:\n2:\n3:\n4:\n  E\nF\n')));

            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            
            clearvars -except test_figure;
            
            fprintf('    Multiline messages.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg =  logging.logger({hnd});
            
            logg.message(sprintf('Hello\nWorld'));
            logg.beg_node('A');
            logg.message(sprintf('Hello\nWorld'));
            logg.end_node();
            
            assert(check.same(hnd.logged_data,sprintf('Hello\nWorld\nA:\n  Hello\n  World\n')));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Functions "beg_node" and "end_node".\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg.beg_node('A');
            
            assert(check.same(logg.level_stack,logging.level.Experiment));
            assert(check.same(logg.per_handler_indent,[2;2;2;2;2]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment}}));

            logg.end_node();
            
            assert(check.same(logg.level_stack,logging.level.Experiment));
            assert(check.same(logg.per_handler_indent,[0;0;0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment}}));
                                            
            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();

            clearvars -except test_figure;
            
            fprintf('  Functions "beg_classifier" and "end_classifier".\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg.beg_classifier('A');
            
            assert(check.same(logg.level_stack,[logging.level.Experiment;logging.level.Classifier]));
            assert(check.same(logg.per_handler_indent,[2;2;2;2;2]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment}}));
                                            
            logg.end_classifier();
            
            assert(check.same(logg.level_stack,[logging.level.Experiment]));
            assert(check.same(logg.per_handler_indent,[0;0;0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment}}));
                                            
            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            
            clearvars -except test_figure;
            
            fprintf('  Functions "beg_regressor" and "end_regressor".\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg.beg_regressor('A');
            
            assert(check.same(logg.level_stack,[logging.level.Experiment;logging.level.Regressor]));
            assert(check.same(logg.per_handler_indent,[2;2;2;2;2]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment}}));
                                            
            logg.end_regressor();
            
            assert(check.same(logg.level_stack,[logging.level.Experiment]));
            assert(check.same(logg.per_handler_indent,[0;0;0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment}}));
                                            
            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            
            clearvars -except test_figure;
            
            fprintf('  Functions "beg_transform" and "end_transform".\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg.beg_transform('A');
            
            assert(check.same(logg.level_stack,[logging.level.Experiment;logging.level.Transform]));
            assert(check.same(logg.per_handler_indent,[2;2;2;2;2]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment}}));
                                            
            logg.end_transform();
            
            assert(check.same(logg.level_stack,[logging.level.Experiment]));
            assert(check.same(logg.per_handler_indent,[0;0;0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment}}));
                                            
            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            
            clearvars -except test_figure;
            
            fprintf('  Functions "beg_results" and "end_results".\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg.beg_results('A');
            
            assert(check.same(logg.level_stack,[logging.level.Experiment;logging.level.Results]));
            assert(check.same(logg.per_handler_indent,[2;2;2;2;2]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment};...
                                                   {logging.level.Experiment;logging.level.Experiment}}));
                                            
            logg.end_results();
            
            assert(check.same(logg.level_stack,[logging.level.Experiment]));
            assert(check.same(logg.per_handler_indent,[0;0;0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment}}));
                                            
            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "new_node".\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg_a =  logg.new_node('A');
            
            assert(check.same(length(logg_a.handlers),5));
            assert(check.same(logg_a.handlers{1}.active,true));
            assert(check.same(logg_a.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg_a.handlers{1}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{2}.active,true));
            assert(check.same(logg_a.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg_a.handlers{2}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{3}.active,true));
            assert(check.same(logg_a.handlers{3}.min_level,logging.level.Regressor));
            assert(check.same(logg_a.handlers{3}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{4}.active,true));
            assert(check.same(logg_a.handlers{4}.min_level,logging.level.Transform));
            assert(check.same(logg_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{5}.active,true));
            assert(check.same(logg_a.handlers{5}.min_level,logging.level.Results));
            assert(check.same(logg_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers_count,5));
            assert(check.same(logg_a.active,true));
            assert(check.same(logg_a.level_stack,[logging.level.Experiment]));
            assert(check.same(logg_a.per_handler_indent,[2;2;2;2;2]));
            assert(check.same(logg_a.level_at_indent,{{logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment}}));
            assert(check.same(length(logg.handlers),5));
            assert(check.same(logg.handlers{1}.active,true));
            assert(check.same(logg.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg.handlers{1}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{2}.active,true));
            assert(check.same(logg.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg.handlers{2}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{3}.active,true));
            assert(check.same(logg.handlers{3}.min_level,logging.level.Regressor));
            assert(check.same(logg.handlers{3}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{4}.active,true));
            assert(check.same(logg.handlers{4}.min_level,logging.level.Transform));
            assert(check.same(logg.handlers{4}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{5}.active,true));
            assert(check.same(logg.handlers{5}.min_level,logging.level.Results));
            assert(check.same(logg.handlers{5}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers_count,5));
            assert(check.same(logg.active,true));
            assert(check.same(logg.level_stack,[logging.level.Experiment]));
            assert(check.same(logg.per_handler_indent,[0;0;0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment}}));

            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();

            clearvars -except test_figure;
            
            fprintf('  Function "new_classifier".\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg_a =  logg.new_classifier('A');
            
            assert(check.same(length(logg_a.handlers),5));
            assert(check.same(logg_a.handlers{1}.active,true));
            assert(check.same(logg_a.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg_a.handlers{1}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{2}.active,true));
            assert(check.same(logg_a.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg_a.handlers{2}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{3}.active,true));
            assert(check.same(logg_a.handlers{3}.min_level,logging.level.Regressor));
            assert(check.same(logg_a.handlers{3}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{4}.active,true));
            assert(check.same(logg_a.handlers{4}.min_level,logging.level.Transform));
            assert(check.same(logg_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{5}.active,true));
            assert(check.same(logg_a.handlers{5}.min_level,logging.level.Results));
            assert(check.same(logg_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers_count,5));
            assert(check.same(logg_a.active,true));
            assert(check.same(logg_a.level_stack,[logging.level.Experiment;logging.level.Classifier]));
            assert(check.same(logg_a.per_handler_indent,[2;2;2;2;2]));
            assert(check.same(logg_a.level_at_indent,{{logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment}}));
            assert(check.same(length(logg.handlers),5));
            assert(check.same(logg.handlers{1}.active,true));
            assert(check.same(logg.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg.handlers{1}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{2}.active,true));
            assert(check.same(logg.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg.handlers{2}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{3}.active,true));
            assert(check.same(logg.handlers{3}.min_level,logging.level.Regressor));
            assert(check.same(logg.handlers{3}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{4}.active,true));
            assert(check.same(logg.handlers{4}.min_level,logging.level.Transform));
            assert(check.same(logg.handlers{4}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{5}.active,true));
            assert(check.same(logg.handlers{5}.min_level,logging.level.Results));
            assert(check.same(logg.handlers{5}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers_count,5));
            assert(check.same(logg.active,true));
            assert(check.same(logg.level_stack,[logging.level.Experiment]));
            assert(check.same(logg.per_handler_indent,[0;0;0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment}}));
                                            
            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "new_regressor".\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg_a =  logg.new_regressor('A');
            
            assert(check.same(length(logg_a.handlers),5));
            assert(check.same(logg_a.handlers{1}.active,true));
            assert(check.same(logg_a.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg_a.handlers{1}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{2}.active,true));
            assert(check.same(logg_a.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg_a.handlers{2}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{3}.active,true));
            assert(check.same(logg_a.handlers{3}.min_level,logging.level.Regressor));
            assert(check.same(logg_a.handlers{3}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{4}.active,true));
            assert(check.same(logg_a.handlers{4}.min_level,logging.level.Transform));
            assert(check.same(logg_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{5}.active,true));
            assert(check.same(logg_a.handlers{5}.min_level,logging.level.Results));
            assert(check.same(logg_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers_count,5));
            assert(check.same(logg_a.active,true));
            assert(check.same(logg_a.level_stack,[logging.level.Experiment;logging.level.Regressor]));
            assert(check.same(logg_a.per_handler_indent,[2;2;2;2;2]));
            assert(check.same(logg_a.level_at_indent,{{logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment}}));
            assert(check.same(length(logg.handlers),5));
            assert(check.same(logg.handlers{1}.active,true));
            assert(check.same(logg.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg.handlers{1}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{2}.active,true));
            assert(check.same(logg.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg.handlers{2}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{3}.active,true));
            assert(check.same(logg.handlers{3}.min_level,logging.level.Regressor));
            assert(check.same(logg.handlers{3}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{4}.active,true));
            assert(check.same(logg.handlers{4}.min_level,logging.level.Transform));
            assert(check.same(logg.handlers{4}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{5}.active,true));
            assert(check.same(logg.handlers{5}.min_level,logging.level.Results));
            assert(check.same(logg.handlers{5}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers_count,5));
            assert(check.same(logg.active,true));
            assert(check.same(logg.level_stack,[logging.level.Experiment]));
            assert(check.same(logg.per_handler_indent,[0;0;0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment}}));
                                            
            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "new_transform".\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg_a =  logg.new_transform('A');
            
            assert(check.same(length(logg_a.handlers),5));
            assert(check.same(logg_a.handlers{1}.active,true));
            assert(check.same(logg_a.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg_a.handlers{1}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{2}.active,true));
            assert(check.same(logg_a.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg_a.handlers{2}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{3}.active,true));
            assert(check.same(logg_a.handlers{3}.min_level,logging.level.Regressor));
            assert(check.same(logg_a.handlers{3}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{4}.active,true));
            assert(check.same(logg_a.handlers{4}.min_level,logging.level.Transform));
            assert(check.same(logg_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{5}.active,true));
            assert(check.same(logg_a.handlers{5}.min_level,logging.level.Results));
            assert(check.same(logg_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers_count,5));
            assert(check.same(logg_a.active,true));
            assert(check.same(logg_a.level_stack,[logging.level.Experiment;logging.level.Transform]));
            assert(check.same(logg_a.per_handler_indent,[2;2;2;2;2]));
            assert(check.same(logg_a.level_at_indent,{{logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment}}));
            assert(check.same(length(logg.handlers),5));
            assert(check.same(logg.handlers{1}.active,true));
            assert(check.same(logg.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg.handlers{1}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{2}.active,true));
            assert(check.same(logg.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg.handlers{2}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{3}.active,true));
            assert(check.same(logg.handlers{3}.min_level,logging.level.Regressor));
            assert(check.same(logg.handlers{3}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{4}.active,true));
            assert(check.same(logg.handlers{4}.min_level,logging.level.Transform));
            assert(check.same(logg.handlers{4}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{5}.active,true));
            assert(check.same(logg.handlers{5}.min_level,logging.level.Results));
            assert(check.same(logg.handlers{5}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers_count,5));
            assert(check.same(logg.active,true));
            assert(check.same(logg.level_stack,[logging.level.Experiment]));
            assert(check.same(logg.per_handler_indent,[0;0;0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment}}));
                                            
            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            
            clearvars -except test_figure;

            fprintf('  Function "new_results".\n');
            
            hnd1 = logging.handlers.testing(logging.level.Experiment);
            hnd2 = logging.handlers.testing(logging.level.Classifier);
            hnd3 = logging.handlers.testing(logging.level.Regressor);
            hnd4 = logging.handlers.testing(logging.level.Transform);
            hnd5 = logging.handlers.testing(logging.level.Results);
            logg =  logging.logger({hnd1,hnd2,hnd3,hnd4,hnd5});
            
            logg_a =  logg.new_results('A');
            
            assert(check.same(length(logg_a.handlers),5));
            assert(check.same(logg_a.handlers{1}.active,true));
            assert(check.same(logg_a.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg_a.handlers{1}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{2}.active,true));
            assert(check.same(logg_a.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg_a.handlers{2}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{3}.active,true));
            assert(check.same(logg_a.handlers{3}.min_level,logging.level.Regressor));
            assert(check.same(logg_a.handlers{3}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{4}.active,true));
            assert(check.same(logg_a.handlers{4}.min_level,logging.level.Transform));
            assert(check.same(logg_a.handlers{4}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers{5}.active,true));
            assert(check.same(logg_a.handlers{5}.min_level,logging.level.Results));
            assert(check.same(logg_a.handlers{5}.logged_data,sprintf('A:\n')));
            assert(check.same(logg_a.handlers_count,5));
            assert(check.same(logg_a.active,true));
            assert(check.same(logg_a.level_stack,[logging.level.Experiment;logging.level.Results]));
            assert(check.same(logg_a.per_handler_indent,[2;2;2;2;2]));
            assert(check.same(logg_a.level_at_indent,{{logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment};...
                                                     {logging.level.Experiment;logging.level.Experiment}}));
            assert(check.same(length(logg.handlers),5));
            assert(check.same(logg.handlers{1}.active,true));
            assert(check.same(logg.handlers{1}.min_level,logging.level.Experiment));
            assert(check.same(logg.handlers{1}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{2}.active,true));
            assert(check.same(logg.handlers{2}.min_level,logging.level.Classifier));
            assert(check.same(logg.handlers{2}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{3}.active,true));
            assert(check.same(logg.handlers{3}.min_level,logging.level.Regressor));
            assert(check.same(logg.handlers{3}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{4}.active,true));
            assert(check.same(logg.handlers{4}.min_level,logging.level.Transform));
            assert(check.same(logg.handlers{4}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers{5}.active,true));
            assert(check.same(logg.handlers{5}.min_level,logging.level.Results));
            assert(check.same(logg.handlers{5}.logged_data,sprintf('A:\n')));
            assert(check.same(logg.handlers_count,5));
            assert(check.same(logg.active,true));
            assert(check.same(logg.level_stack,[logging.level.Experiment]));
            assert(check.same(logg.per_handler_indent,[0;0;0;0;0]));
            assert(check.same(logg.level_at_indent,{{logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment};...
                                                   {logging.level.Experiment}}));
                                            
            logg.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            hnd4.close();
            hnd5.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "close".\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg =  logging.logger({hnd});

            logg.close();
            
            assert(check.same(logg.active,false));
            
            logg.close();
            
            assert(check.same(logg.active,false));
            
            hnd.close();
            
            clearvars -except test_figure;
        end
    end
end
