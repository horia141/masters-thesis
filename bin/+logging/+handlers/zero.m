classdef zero < logging.handler
    methods (Access=public)
        function [obj] = zero(min_level)
            assert(tc.scalar(min_level));
            assert(tc.logging_level(min_level));

            obj = obj@logging.handler(min_level);
        end
    end
    
    methods (Access=protected)
        function [] = do_send(~,~)
        end
        
        function [] = do_close(~)
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logging.handlers.zero".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With minimum level "Architecture".\n');
            
            hnd = logging.handlers.zero(logging.level.Architecture);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Architecture);
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With minimum level "Results".\n');
            
            hnd = logging.handlers.zero(logging.level.Results);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Results);
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With minimum level "Dataset_IO".\n');
            
            hnd = logging.handlers.zero(logging.level.Dataset_IO);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Dataset_IO);
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "send".\n');
            
            hnd = logging.handlers.zero(logging.level.TopLevel);
            
            hnd.send('Successful send of message.\n');
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "close".\n');
            
            hnd = logging.handlers.zero(logging.level.TopLevel);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            clearvars -except display;
        end
    end
end