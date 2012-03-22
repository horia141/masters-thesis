classdef sendmail < logging.handler
    properties (GetAccess=public,SetAccess=immutable)
        email_addrs;
        email_addrs_count;
        sender;
        smtp_server;
    end
    
    methods (Access=public)
        function [obj] = sendmail(email_addrs,min_level,sender,smtp_server)
            assert(tc.vector(email_addrs) && tc.cell(email_addrs) && ...
                   tc.check(cellfun(@(c)tc.scalar(c) && tc.string(c),email_addrs)));
            assert(tc.scalar(min_level) && tc.logging_level(min_level));
            assert(~exist('sender','var') || (tc.scalar(sender) && tc.string(sender)));
            assert(~exist('smtp_server','var') || (tc.scalar(sender) && tc.string(smtp_server)));

            if exist('sender','var')
                sender_t = sender;
            else
                sender_t = 'coman@inb.uni-luebeck.de';
            end

            if exist('smtp_server','var')
                smtp_server_t = smtp_server;
            else
                smtp_server_t = 'pc07.inb.uni-luebeck.de';
            end

            obj = obj@logging.handler(min_level);
            obj.email_addrs = utils.force_col(email_addrs);
            obj.email_addrs_count = length(email_addrs);
            obj.sender = sender_t;
            obj.smtp_server = smtp_server_t;
        end
    end
    
    methods (Access=protected)
        function [] = do_send(obj,message)
            saved_sender = getpref('Internet','E_mail');
            saved_smtp_server = getpref('Internet','SMTP_Server');
            
            setpref('Internet','E_mail',obj.sender);            
            setpref('Internet','SMTP_Server',obj.smtp_server);

            sendmail(obj.email_addrs,'master log',[message 10]);
            
            setpref('Internet','E_mail',saved_sender);
            setpref('Internet','SMTP_Server',saved_smtp_server);
        end
        
        function [] = do_close(~)
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "logging.handlers.sendmail"\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With minimum level "Details" and no optional parameters.\n');
            
            hnd = logging.handlers.sendmail({'coman@inb.uni-luebeck.de'},logging.level.Details);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Details);
            assert(length(hnd.email_addrs) == 1);
            assert(strcmp(hnd.email_addrs{1},'coman@inb.uni-luebeck.de'));
            assert(hnd.email_addrs_count == 1);
            assert(strcmp(hnd.sender,'coman@inb.uni-luebeck.de'));
            assert(strcmp(hnd.smtp_server,'pc07.inb.uni-luebeck.de'));
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With minimum level "Status" and no optional parameters.\n');
            
            hnd = logging.handlers.sendmail({'coman@inb.uni-luebeck.de'},logging.level.Status);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Status);
            assert(length(hnd.email_addrs) == 1);
            assert(strcmp(hnd.email_addrs{1},'coman@inb.uni-luebeck.de'));
            assert(hnd.email_addrs_count == 1);
            assert(strcmp(hnd.sender,'coman@inb.uni-luebeck.de'));
            assert(strcmp(hnd.smtp_server,'pc07.inb.uni-luebeck.de'));
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With minimum level "Error" and no optional parameters.\n');
            
            hnd = logging.handlers.sendmail({'coman@inb.uni-luebeck.de'},logging.level.Error);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Error);
            assert(length(hnd.email_addrs) == 1);
            assert(strcmp(hnd.email_addrs{1},'coman@inb.uni-luebeck.de'));
            assert(hnd.email_addrs_count == 1);
            assert(strcmp(hnd.sender,'coman@inb.uni-luebeck.de'));
            assert(strcmp(hnd.smtp_server,'pc07.inb.uni-luebeck.de'));
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With minimum level "Status" and "sender" specified.\n');
            
            hnd = logging.handlers.sendmail({'coman@inb.uni-luebeck.de'},logging.level.Error,'comanAA@inb.uni-luebeck.de');
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Error);
            assert(length(hnd.email_addrs) == 1);
            assert(strcmp(hnd.email_addrs{1},'coman@inb.uni-luebeck.de'));
            assert(hnd.email_addrs_count == 1);
            assert(strcmp(hnd.sender,'comanAA@inb.uni-luebeck.de'));
            assert(strcmp(hnd.smtp_server,'pc07.inb.uni-luebeck.de'));
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With minimum level "Status" and "sender" and "smtp_server" specified.\n');
            
            hnd = logging.handlers.sendmail({'coman@inb.uni-luebeck.de'},logging.level.Error,'comanBB@inb.uni-luebeck.de','pc07AA.inb.uni-luebeck.de');
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Error);
            assert(length(hnd.email_addrs) == 1);
            assert(strcmp(hnd.email_addrs{1},'coman@inb.uni-luebeck.de'));
            assert(hnd.email_addrs_count == 1);
            assert(strcmp(hnd.sender,'comanBB@inb.uni-luebeck.de'));
            assert(strcmp(hnd.smtp_server,'pc07AA.inb.uni-luebeck.de'));
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With multiple addresses specified.\n');
            
            hnd = logging.handlers.sendmail({'coman@inb.uni-luebeck.de' 'comanAA@inb.uni-luebeck.de'},logging.level.Error);
            
            assert(hnd.active == true);
            assert(hnd.min_level == logging.level.Error);
            assert(length(hnd.email_addrs) == 2);
            assert(strcmp(hnd.email_addrs{1},'coman@inb.uni-luebeck.de'));
            assert(strcmp(hnd.email_addrs{2},'comanAA@inb.uni-luebeck.de'));
            assert(hnd.email_addrs_count == 2);
            assert(strcmp(hnd.sender,'coman@inb.uni-luebeck.de'));
            assert(strcmp(hnd.smtp_server,'pc07.inb.uni-luebeck.de'));
            
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "send".\n');
            
            hnd = logging.handlers.stdout(logging.level.Details);
            
            %hnd.send('Successful send of message.\n');
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
