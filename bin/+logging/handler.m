classdef handler < handle
    properties (GetAccess=public,SetAccess=immutable)
        min_level;
    end

    properties (GetAccess=public,SetAccess=private)
        active;
    end

    methods (Access=public)
        function [obj] = handler(min_level)
            assert(check.scalar(min_level));
            assert(check.logging_level(min_level));

            obj.min_level = min_level;
            obj.active = true;
        end
        
        function [] = send(obj,message)
            assert(check.scalar(obj));
            assert(check.logging_handler(obj));
            assert(obj.active);
            assert(check.scalar(message))
            assert(check.string(message));
            
            obj.do_send(message);
        end
        
        function [] = close(obj)
            assert(check.scalar(obj));
            assert(check.logging_handler(obj));

            if obj.active
                obj.do_close();            
                obj.active = false;
            end
        end
        
        function [] = delete(obj)
            assert(check.scalar(obj));
            assert(check.logging_handler(obj));

            obj.close();
        end
    end
    
    methods (Abstract,Access=protected)
        do_send(obj,message);
        do_close(obj);
    end
end
