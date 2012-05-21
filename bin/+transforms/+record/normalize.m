classdef normalize < transform
    properties (GetAccess=public,SetAccess=immutable)
        kept_mean;
    end
    
    methods (Access=public)
        function [obj] = normalize(train_sample_plain,logger)
            assert(tc.dataset_record(train_sample_plain));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            logger.message('Computing sample mean.');
            
            kept_mean_t = mean(train_sample_plain,2);
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = input_geometry;
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.kept_mean = kept_mean_t;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            logger.message('Normalizing sample.');
            
            sample_coded_t1 = bsxfun(@minus,sample_plain,obj.kept_mean);
            sample_coded = bsxfun(@rdivide,sample_coded_t1,sqrt(sum(sample_coded_t1 .^ 2,1)));
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.record.normalize".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0; 0 0.3],10000)';
            
            t = transforms.record.normalize(s,log);
    
            assert(tc.same(t.kept_mean,[3;3],'Epsilon',0.1));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,2));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing sample mean.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0; 0 0.3],10000)';
            
            t = transforms.record.normalize(s,log);
            s_p = t.code(s,log);
            
            % Use "mean(s,2)" instead of [3;3] because of precision problems.
            assert(tc.same(s_p,(s - repmat(mean(s,2),1,10000)) ./ repmat(sqrt(sum((s - repmat(mean(s,2),1,10000)) .^ 2,1)),2,1),'Epsilon',0.1));
            assert(tc.same(mean(s_p,2),[0;0],'Epsilon',0.1));
            assert(tc.same(sqrt(sum(s_p .^ 2,1)),ones(1,10000),'Epsilon',0.1));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing sample mean.\n',...
                                                          'Normalizing sample.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original sample.');
                subplot(1,2,2);
                scatter(s_p(1,:),s_p(2,:),'x');
                axis([-4 6 -4 6]);
                axis('square');
                title('Normalized sample.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  With a more complex sample.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([3 3],[0.1 0; 0 0.03],10000);
                 mvnrnd([1 1],[0.1 0; 0 0.03],10000)]';

            t = transforms.record.normalize(s,log);
            s_p = t.code(s,log);
            
            % Use "mean(s,2)" instead of [3;3] because of precision problems.
            assert(tc.same(s_p,(s - repmat(mean(s,2),1,20000)) ./ repmat(sqrt(sum((s - repmat(mean(s,2),1,20000)) .^ 2,1)),2,1),'Epsilon',0.1));
            assert(tc.same(mean(s_p,2),[0;0],'Epsilon',0.1));
            assert(tc.same(sqrt(sum(s_p .^ 2,1)),ones(1,20000),'Epsilon',0.1));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                hold('on');
                scatter(s(1,1:10000),s(2,1:10000),'r','o');
                scatter(s(1,10001:20000),s(2,10001:20000),'b','o');
                hold('off');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original sample.');
                subplot(1,2,2);
                hold('on');
                scatter(s_p(1,1:10000),s_p(2,1:10000),'r','x');
                scatter(s_p(1,10001:20000),s_p(2,10001:20000),'b','x');
                hold('off');
                axis([-4 6 -4 6]);
                axis('square');
                title('Normalized sample.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  With more complex instances.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50)];
            
            t = transforms.record.normalize(s,log);
            s_p = t.code(s,log);
            
            % Use "mean(s,2)" instead of [3;3] because of precision problems.
            assert(tc.same(s_p,(s - repmat(mean(s,2),1,4)) ./ repmat(sqrt(sum((s - repmat(mean(s,2),1,4)) .^ 2,1)),50,1),'Epsilon',0.1));
            assert(tc.same(mean(s_p,2),zeros(50,1),'Epsilon',0.1));
            assert(tc.same(sqrt(sum(s_p .^ 2,1)),ones(1,4),'Epsilon',0.1));
            
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
