classdef file < logging.handler
    properties (GetAccess=public,SetAccess=immutable)
        file_id;
    end
    
    methods (Access=public)
        function [obj] = file(logging_file_path,min_level)
            assert(check.scalar(logging_file_path));
            assert(check.string(logging_file_path));
            assert(check.scalar(min_level));
            assert(check.logging_level(min_level));
            
            [file_id_t,file_msg] = fopen(logging_file_path,'wt');
            
            if file_id_t == -1
                throw(MException('master:NoOpen',...
                         sprintf('Could not open logging file "%s": %s!',logging_file_path,file_msg)));
            end
            
            obj = obj@logging.handler(min_level);
            obj.file_id = file_id_t;
        end
    end
    
    methods (Access=protected)
        function [] = do_send(obj,message)
            fprintf(obj.file_id,message);
        end
        
        function [] = do_close(obj)
            fclose(obj.file_id);
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logging.handlers.file".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With good logging path and minimum level "Experiment".\n');
            
            hnd = logging.handlers.file('../test/log1.log',logging.level.Experiment);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Experiment);
            
            hnd.close();
            
            [ls_code,~] = system('ls ../test/log1.log');
            rm_code = system('rm ../test/log1.log');
            
            assert(ls_code == 0);
            assert(rm_code == 0);
            
            clearvars -except test_figure;
            
            fprintf('    With good logging path and minimum level "Results".\n');
            
            hnd = logging.handlers.file('../test/log1.log',logging.level.Results);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Results);
            
            hnd.close();
            
            [ls_code,~] = system('ls ../test/log1.log');
            rm_code = system('rm ../test/log1.log');
            
            assert(ls_code == 0);
            assert(rm_code == 0);
            
            clearvars -except test_figure;
            
            fprintf('    With invalid external input.\n');
            
            try
                touch_code = system('touch ../test/log1.log');
                chmod_code = system('chmod a-w ../test/log1.log');
                
                assert(touch_code == 0);
                assert(chmod_code == 0);
                
                logging.handlers.file('../test/log1.log',logging.level.Experiment);
                
                chmod2_code = system('chmod a+w ../test/log1.log');
                rm_code = system('rm ../test/log1.log');
                
                assert(chmod2_code == 0);
                assert(rm_code == 0);
                assert(false);
            catch exp
                chmod2_code = system('chmod a+w ../test/log1.log');
                rm_code = system('rm ../test/log1.log');
                
                assert(chmod2_code == 0);
                assert(rm_code == 0);
                
                if strcmp(exp.message,'Could not open logging file "../test/log1.log": Permission denied!')
                    fprintf('      Passes "Permission denied!" test.\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except test_figure;
            
            fprintf('  Function "send".\n');
            
            hnd = logging.handlers.file('../test/log1.log',logging.level.Experiment);
            
            hnd.send('Successful send of message.\n');
            hnd.close();
            
            [cat_code,cat_res] = system('cat ../test/log1.log');
            rm_code = system('rm ../test/log1.log');
            
            assert(cat_code == 0);
            assert(strcmp(cat_res,sprintf('%s\n','Successful send of message.')));
            assert(rm_code == 0);
            
            clearvars -except test_figure;
            
            fprintf('  Function "close".\n');
            
            hnd = logging.handlers.file('../test/log1.log',logging.level.Experiment);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            rm_code = system('rm ../test/log1.log');
            
            assert(rm_code == 0);
            
            clearvars -except test_figure;
        end
    end
end
