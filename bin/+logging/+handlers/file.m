classdef file < logging.handler
    properties (GetAccess=public,SetAccess=immutable)
        file_id;
    end
    
    methods (Access=public)
        function [obj] = file(logging_file_path,min_level)
            assert(tc.scalar(logging_file_path));
            assert(tc.string(logging_file_path));
            assert(tc.scalar(min_level));
            assert(tc.logging_level(min_level));
            
            [file_id_t,file_msg] = fopen(logging_file_path,'wt');
            
            if file_id_t == -1
                throw(MException('master:logging:handler:file:NoOpen',...
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
            
            fprintf('    With good logging path and minimum level "Architecture".\n');
            
            hnd = logging.handlers.file('../data/log1.log',logging.level.Architecture);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Architecture);
            
            hnd.close();
            
            [ls_code,~] = system('ls ../data/log1.log');
            [rm_code,~] = system('rm ../data/log1.log');
            
            assert(ls_code == 0);
            assert(rm_code == 0);
            
            clearvars -except display;
            
            fprintf('    With good logging path and minimum level "Results".\n');
            
            hnd = logging.handlers.file('../data/log1.log',logging.level.Results);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Results);
            
            hnd.close();
            
            [ls_code,~] = system('ls ../data/log1.log');
            [rm_code,~] = system('rm ../data/log1.log');
            
            assert(ls_code == 0);
            assert(rm_code == 0);
            
            clearvars -except display;
            
            fprintf('    With good logging path and minimum level "Dataset_IO".\n');
            
            hnd = logging.handlers.file('../data/log1.log',logging.level.Dataset_IO);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Dataset_IO);
            
            hnd.close();
            
            [ls_code,~] = system('ls ../data/log1.log');
            [rm_code,~] = system('rm ../data/log1.log');
            
            assert(ls_code == 0);
            assert(rm_code == 0);
            
            clearvars -except display;
            
            fprintf('    With invalid external input.\n');
            
            try
                !touch ../data/log1.log
                !chmod a-w ../data/log1.log
                hnd = logging.handlers.file('../data/log1.log',logging.level.TopLevel);
                !chmod a+w ../data/log1.log
                !rm ../data/log1.log
                assert(false);
            catch exp
                !chmod a+w ../data/log1.log
                !rm ../data/log1.log
                
                if strcmp(exp.message,'Could not open logging file "../data/log1.log": Permission denied!')
                    fprintf('      Passes "Permission denied!" test.\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except display;
            
            fprintf('  Function "send".\n');
            
            hnd = logging.handlers.file('../data/log1.log',logging.level.TopLevel);
            
            hnd.send('Successful send of message.\n');
            hnd.close();
            
            [cat_code,cat_res] = system('cat ../data/log1.log');
            [rm_code,~] = system('rm ../data/log1.log');
            
            assert(cat_code == 0);
            assert(strcmp(cat_res,sprintf('%s\n','Successful send of message.')));
            assert(rm_code == 0);
            
            clearvars -except display;
            
            fprintf('  Function "close".\n');
            
            hnd = logging.handlers.file('../data/log1.log',logging.level.TopLevel);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            [rm_code,~] = system('rm ../data/log1.log');
            
            assert(rm_code == 0);
            
            clearvars -except display;
        end
    end
end
