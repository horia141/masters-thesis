classdef logger < handle
    properties (GetAccess=public,SetAccess=immutable)
        handlers;
        handlers_count;
    end
    
    properties (GetAccess=public,SetAccess=private)
        active;
        node_indent;
        node_indent_level;
        curr_level;
    end
    
    methods (Access=public)
        function [obj] = logger(varargin)
            assert(tc.vector(varargin) && tc.cell(varargin) && ...
                   tc.check(cellfun(@(c)tc.scalar(c) && tc.logging_handler(c) && (c.active),varargin)));
            
            obj.handlers = varargin';
            obj.handlers_count = length(varargin);
            obj.active = true;
            obj.node_indent = zeros(length(varargin),1);
            obj.node_indent_level = arrayfun(@(i){logging.level.Status},1:length(varargin),'UniformOutput',false)';
            obj.curr_level = logging.level.Status;
        end
        
        function [] = beg_node(obj,message_fmt,varargin)
            assert(obj.active == true);
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || ...
                    (tc.vector(varargin) && tc.cell(varargin) && ...
                     tc.check(cellfun(@(c)tc.string(c) || (tc.scalar(c) && tc.value(c)),varargin))));
               
            message = sprintf(message_fmt,varargin{:});
            
            for i = 1:obj.handlers_count
                if obj.curr_level >= obj.handlers{i}.min_level
                    message_spec = sprintf('%s%s:\n',repmat(' ',1,obj.node_indent(i)),message);
                    obj.node_indent(i) = obj.node_indent(i) + 2;
                    obj.node_indent_level{i} = cat(1,obj.node_indent_level{i},{obj.curr_level});
                    obj.handlers{i}.send(message_spec);
                end
            end
        end
        
        function [] = end_node(obj)
            assert(obj.active == true);
            
            for i = 1:obj.handlers_count
                if obj.curr_level >= obj.handlers{i}.min_level
                    assert(obj.curr_level == obj.node_indent_level{i}{end});
                    
                    obj.node_indent(i) = obj.node_indent(i) - 2;
                    obj.node_indent_level{i} = obj.node_indent_level{i}(1:end-1);
                end
            end
        end
        
        function [] = message(obj,message_fmt,varargin)
            assert(obj.active == true);
            assert(tc.string(message_fmt));
            assert(tc.empty(varargin) || ...
                    (tc.vector(varargin) && tc.cell(varargin) && ...
                     tc.check(cellfun(@(c)tc.string(c) || (tc.scalar(c) && tc.value(c)),varargin))));
               
            message = sprintf(message_fmt,varargin{:});
            
            for i = 1:obj.handlers_count
                if obj.curr_level >= obj.handlers{i}.min_level
                    message_spec = sprintf('%s%s\n',repmat(' ',1,obj.node_indent(i)),message);
                    obj.handlers{i}.send(message_spec);
                end
            end
        end
        
        function [] = beg_details(obj)
            assert(obj.active == true);
            assert(obj.curr_level == logging.level.Status);
            
            obj.curr_level = logging.level.Details;
        end
        
        function [] = end_details(obj)
            assert(obj.active == true);
            assert(obj.curr_level == logging.level.Details);
            
            obj.curr_level = logging.level.Status;
        end
        
        function [] = beg_error(obj)
            assert(obj.active == true);
            assert(obj.curr_level == logging.level.Status);
            
            obj.curr_level = logging.level.Error;
        end
        
        function [] = end_error(obj)
            assert(obj.active == true);
            assert(obj.curr_level == logging.level.Error);
            
            obj.curr_level = logging.level.Status;
        end
        
        function [] = close(obj)
            assert(obj.curr_level == logging.level.Status);
            assert(tc.check(cellfun(@(c)tc.scalar(c) && (c{1} == logging.level.Status),obj.node_indent_level)));

            if obj.active
                obj.active = false;
            end
        end
        
        function [] = delete(obj)
            if (obj.curr_level == logging.level.Status) && ...
               (tc.check(cellfun(@(c)tc.scalar(c) && (c{1} == logging.level.Status),obj.node_indent_level)))
                obj.close();
            else
                % It is not safe to close the object in this state. Rather
                % than assert in a destructor, we'll let it slide.
            end
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logger".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd1 = logging.handlers.file('../data/log1.log',logging.level.Details);
            hnd2 = logging.handlers.file('../data/log2.log',logging.level.Status);
            hnd3 = logging.handlers.file('../data/log3.log',logging.level.Error);
            log = logging.logger(hnd1,hnd2,hnd3);
            
            assert(log.handlers{1} == hnd1);
            assert(log.handlers{2} == hnd2);
            assert(log.handlers{3} == hnd3);
            assert(tc.check(log.node_indent == [0;0;0]));
            assert(tc.check(cellfun(@(c)tc.scalar(c) && (c{1} == logging.level.Status),log.node_indent_level)));
            assert(log.curr_level == logging.level.Status);
            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();            
            
            !rm ../data/log1.log
            !rm ../data/log2.log
            !rm ../data/log3.log
            
            clearvars -except display;
            
            fprintf('  Complex logging.\n');
            
            hnd1 = logging.handlers.file('../data/log1.log',logging.level.Details);
            hnd2 = logging.handlers.file('../data/log2.log',logging.level.Status);
            hnd3 = logging.handlers.file('../data/log3.log',logging.level.Error);
            log = logging.logger(hnd1,hnd2,hnd3);
            
            log.message('A');
            log.message('B');
            log.beg_node('n_C');
            log.message('D');
            log.message('E');
            log.end_node();
            log.beg_details();
            log.beg_node('n_f');
            log.message('aaa %d',10);
            log.message('bbb %.3f',10.75);
            log.end_node();
            log.end_details();
            log.beg_error();
            log.message('ALERT %s%d','error',10);
            log.end_error();
            log.beg_node('AFT');
            log.message('OLA');
            log.beg_node('Koral');
            log.beg_details();
            log.beg_node('Mbasa');
            log.message('Test');
            log.message('Test2');
            log.end_node();
            log.end_details();
            log.beg_node('Testing %s %d %s','AAA',10,'BBB');
            log.beg_error();
            log.message('ALERT2');
            log.end_error();
            log.end_node();
            log.end_node();
            log.end_node();
            
            log.close();
            hnd1.close();
            hnd2.close();
            hnd3.close();
            
            [cat1_code,cat1_res] = system('cat ../data/log1.log');
            [cat2_code,cat2_res] = system('cat ../data/log2.log');
            [cat3_code,cat3_res] = system('cat ../data/log3.log');
            
            assert(cat1_code == 0);
            assert(strcmp(cat1_res,sprintf(strcat('A\n',...
                                                  'B\n',...
                                                  'n_C:\n',...
                                                  '  D\n',...
                                                  '  E\n',...
                                                  'n_f:\n',...
                                                  '  aaa 10\n',...
                                                  '  bbb 10.750\n',...
                                                  'ALERT error10\n',...
                                                  'AFT:\n',...
                                                  '  OLA\n',...
                                                  '  Koral:\n',...
                                                  '    Mbasa:\n',...
                                                  '      Test\n',...
                                                  '      Test2\n',...
                                                  '    Testing AAA 10 BBB:\n',...
                                                  '      ALERT2\n'))));
            assert(cat2_code == 0);
            assert(strcmp(cat2_res,sprintf(strcat('A\n',...
                                                  'B\n',...
                                                  'n_C:\n',...
                                                  '  D\n',...
                                                  '  E\n',...
                                                  'ALERT error10\n',...
                                                  'AFT:\n',...
                                                  '  OLA\n',...
                                                  '  Koral:\n',...
                                                  '    Testing AAA 10 BBB:\n',...
                                                  '      ALERT2\n'))));
            assert(cat3_code == 0);
            assert(strcmp(cat3_res,sprintf(strcat('ALERT error10\n',...
                                                  'ALERT2\n'))));
                                              
            !rm ../data/log1.log
            !rm ../data/log2.log
            !rm ../data/log3.log
            
            clearvars -except display;
        end
    end
end
