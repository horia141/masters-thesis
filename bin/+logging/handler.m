classdef handler < handle
    properties (GetAccess=public,SetAccess=immutable)
        min_level;
    end

    properties (GetAccess=public,SetAccess=private)
        active;
    end

    methods (Access=public)
        function [obj] = handler(min_level)
            assert(tc.scalar(min_level));
            assert(tc.logging_level(min_level));

            obj.min_level = min_level;
            obj.active = true;
        end
        
        function [] = send(obj,message)
            assert(tc.scalar(obj));
            assert(tc.logging_handler(obj));
            assert(obj.active);
            assert(tc.scalar(message))
            assert(tc.string(message));
            
            obj.do_send(message);
        end
        
        function [] = close(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_handler(obj));

            if obj.active
                obj.do_close();            
                obj.active = false;
            end
        end
        
        function [] = delete(obj)
            assert(tc.scalar(obj));
            assert(tc.logging_handler(obj));

            obj.close();
        end
    end
    
    methods (Abstract,Access=protected)
        do_send(obj,message);
        do_close(obj);
    end
end
