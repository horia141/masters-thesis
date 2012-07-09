classdef testing < logging.handler
    properties (GetAccess=public,SetAccess=private)
        logged_data;
    end
    
    methods (Access=public)
        function [obj] = testing(min_level)
            assert(check.scalar(min_level));
            assert(check.logging_level(min_level));
            
            obj = obj@logging.handler(min_level);
            obj.logged_data = '';
        end
    end
    
    methods (Access=protected)
        function [] = do_send(obj,message)
            obj.logged_data = sprintf('%s%s',obj.logged_data,sprintf(message));
        end
        
        function [] = do_close(~)
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logging.handlers.testing".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With minimum level "Experiment".\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Experiment);
            assert(strcmp(hnd.logged_data,''));
            
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With minimum level "Results".\n');
            
            hnd = logging.handlers.testing(logging.level.Results);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Results);
            assert(strcmp(hnd.logged_data,''));
            
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "send".\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            
            hnd.send('Successful send of message.\n');
            hnd.close();
            
            assert(strcmp(hnd.logged_data,sprintf('Successful send of message.\n')));
            
            clearvars -except test_figure;
            
            fprintf('  Function "close".\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            clearvars -except test_figure;
        end
    end
end
