classdef stdout < logging.handler
    methods (Access=public)
        function [obj] = stdout(min_level)
            assert(tc.scalar(min_level) && tc.logging_level(min_level));

            obj = obj@logging.handler(min_level);
        end
    end
    
    methods (Access=protected)
        function [] = do_send(~,message)
            fprintf(1,message);
        end
        
        function [] = do_close(~)
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logging.handlers.stdout".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With minimum level "Details".\n');
            
            hnd = logging.handlers.stdout(logging.level.Details);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Details);
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With minimum level "Status".\n');
            
            hnd = logging.handlers.stdout(logging.level.Status);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Status);
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With minimum level "Error".\n');
            
            hnd = logging.handlers.stdout(logging.level.Error);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Error);
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "send".\n');
            
            hnd = logging.handlers.stdout(logging.level.Details);
            
            hnd.send('    Successful send of "Details" level message.\n');
            hnd.send('    Successful send of "Status" level message.\n');
            hnd.send('    Successful send of "Error" level message.\n');
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "close".\n');
            
            hnd = logging.handlers.stdout(logging.level.Status);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            clearvars -except display;
        end
    end
end
