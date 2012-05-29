classdef dc_offset < transform
    methods (Access=public)
        function [obj] = dc_offset(train_sample_plain,logger)
            assert(tc.dataset_record(train_sample_plain));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = input_geometry;

            obj = obj@transform(input_geometry,output_geometry,logger);
        end
    end
    
    methods (Access=protected)        
        function [sample_coded] = do_code(~,sample_plain,logger)
            logger.message('Substracting DC component from each sample.');

            sample_coded = bsxfun(@minus,sample_plain,mean(sample_plain,1));
        end
    end

    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.record.dc_offset".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50)];
            
            t = transforms.record.dc_offset(s,log);
            
            assert(tc.same(t.input_geometry,50));
            assert(tc.same(t.output_geometry,50));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50)];
            
            t = transforms.record.dc_offset(s,log);
            s_p = t.code(s,log);
            
            assert(tc.same(s_p,s - repmat(mean(s,1),50,1)));
            assert(tc.same(mean(s_p,1),zeros(1,4)));
            
            if exist('display','var') && (display == true)
                figure();
                for ii = 1:4
                    subplot(4,2,(ii - 1)*2 + 1);
                    plot(s(:,ii));
                    axis([1 50 -5 5]);
                    subplot(4,2,(ii - 1)*2 + 2);
                    plot(s_p(:,ii));
                    axis([1 50 -5 5]);
                end
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
