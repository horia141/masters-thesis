classdef sendmail < logging.handler
    properties (GetAccess=public,SetAccess=immutable)
        title;
        email_addrs;
        email_addrs_count;
        sender;
        smtp_server;
    end
    
    properties (GetAccess=public,SetAccess=private)
        logged_data;
    end
    
    methods (Access=public)
        function [obj] = sendmail(title,email_addrs,sender,smtp_server,min_level)
            assert(check.scalar(title));
            assert(check.string(title));
            assert(check.vector(email_addrs));
            assert(check.cell(email_addrs));
            assert(check.checkf(@check.scalar,email_addrs));
            assert(check.checkf(@check.string,email_addrs));
            assert(check.scalar(sender));
            assert(check.string(sender));
            assert(check.scalar(smtp_server));
            assert(check.string(smtp_server));
            assert(check.scalar(min_level));
            assert(check.logging_level(min_level));

            obj = obj@logging.handler(min_level);
            obj.title = title;
            obj.email_addrs = utils.common.force_col(email_addrs);
            obj.email_addrs_count = length(email_addrs);
            obj.sender = sender;
            obj.smtp_server = smtp_server;
            obj.logged_data = '';
        end
    end
    
    methods (Access=protected)
        function [] = do_send(obj,message)
            obj.logged_data = sprintf('%s%s',obj.logged_data,sprintf(message));
        end
        
        function [] = do_close(obj)
            if ~strcmp(obj.smtp_server,'no-send-mock-server') && ...
               ~check.empty(obj.logged_data)
                attach_file_path = tempname;
                attach_file_fid = fopen(attach_file_path,'wt');
                fprintf(attach_file_fid,obj.logged_data);
                fclose(attach_file_fid);

                saved_sender = getpref('Internet','E_mail');
                saved_smtp_server = getpref('Internet','SMTP_Server');
            
                setpref('Internet','E_mail',obj.sender);            
                setpref('Internet','SMTP_Server',obj.smtp_server);

                sendmail(obj.email_addrs,obj.title,'',attach_file_path);

                setpref('Internet','E_mail',saved_sender);
                setpref('Internet','SMTP_Server',saved_smtp_server);

                ret_code = system(sprintf('rm %s',attach_file_path));

                assert(ret_code == 0);
            end
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logging.handlers.sendmail"\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With minimum level "Experiment".\n');
            
            hnd = logging.handlers.sendmail('Test mail',{'coman@inb.uni-luebeck.de'},'coman@inb.uni-luebeck.de','no-send-mock-server',logging.level.Experiment);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Experiment);
            assert(strcmp(hnd.title,'Test mail'));
            assert(length(hnd.email_addrs) == 1);
            assert(strcmp(hnd.email_addrs{1},'coman@inb.uni-luebeck.de'));
            assert(hnd.email_addrs_count == 1);
            assert(strcmp(hnd.sender,'coman@inb.uni-luebeck.de'));
            assert(strcmp(hnd.smtp_server,'no-send-mock-server'));
            assert(strcmp(hnd.logged_data,''));
            
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With minimum level "Results".\n');
            
            hnd = logging.handlers.sendmail('Test mail',{'coman@inb.uni-luebeck.de'},'coman@inb.uni-luebeck.de','no-send-mock-server',logging.level.Results);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Results);
            assert(strcmp(hnd.title,'Test mail'));
            assert(length(hnd.email_addrs) == 1);
            assert(strcmp(hnd.email_addrs{1},'coman@inb.uni-luebeck.de'));
            assert(hnd.email_addrs_count == 1);
            assert(strcmp(hnd.sender,'coman@inb.uni-luebeck.de'));
            assert(strcmp(hnd.smtp_server,'no-send-mock-server'));
            assert(strcmp(hnd.logged_data,''));
            
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With multiple addresses specified.\n');
            
            hnd = logging.handlers.sendmail('Test mail',{'coman@inb.uni-luebeck.de' 'comanAA@inb.uni-luebeck.de'},'coman@inb.uni-luebeck.de','no-send-mock-server',logging.level.Experiment);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Experiment);
            assert(strcmp(hnd.title,'Test mail'));
            assert(length(hnd.email_addrs) == 2);
            assert(strcmp(hnd.email_addrs{1},'coman@inb.uni-luebeck.de'));
            assert(strcmp(hnd.email_addrs{2},'comanAA@inb.uni-luebeck.de'));
            assert(hnd.email_addrs_count == 2);
            assert(strcmp(hnd.sender,'coman@inb.uni-luebeck.de'));
            assert(strcmp(hnd.smtp_server,'no-send-mock-server'));
            
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "send".\n');
            
            hnd = logging.handlers.sendmail('Test mail',{'coman@inb.uni-luebeck.de'},'coman@inb.uni-luebeck.de','no-send-mock-server',logging.level.Experiment);
            
            hnd.send('Successful send of message.\n');
            hnd.close();
            
            assert(strcmp(hnd.logged_data,sprintf('Successful send of message.\n')));
            
            clearvars -except test_figure;
            
            fprintf('  Function "close".\n');
            
            hnd = logging.handlers.sendmail('Test mail',{'coman@inb.uni-luebeck.de'},'coman@inb.uni-luebeck.de','no-send-mock-server',logging.level.Experiment);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            hnd.close();
            
            assert(hnd.active == false);
            
            clearvars -except test_figure;
        end
    end
end
