classdef stdout < logging.handler
    methods (Access=public)
        function [obj] = stdout(min_level)
            assert(check.scalar(min_level));
            assert(check.logging_level(min_level));

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
            
            fprintf('    With minimum level "Experiment".\n');
            
            hnd = logging.handlers.stdout(logging.level.Experiment);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Experiment);
            
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With minimum level "Results".\n');
            
            hnd = logging.handlers.stdout(logging.level.Results);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Results);
            
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "send".\n');
            
            hnd = logging.handlers.stdout(logging.level.Experiment);
            
            hnd.send('    Successful send of message.\n');
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "close".\n');
            
            hnd = logging.handlers.stdout(logging.level.Experiment);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            clearvars -except test_figure;
        end
    end
end
