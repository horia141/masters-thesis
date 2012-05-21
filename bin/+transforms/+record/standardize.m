classdef standardize < transforms.reversible
    properties (GetAccess=public,SetAccess=immutable)
        kept_mean;
        kept_deviation;
    end
    
    methods (Access=public)
        function [obj] = standardize(train_sample_plain,logger)
            assert(tc.dataset_record(train_sample_plain));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            logger.message('Computing sample mean.');
            
            kept_mean_t = mean(train_sample_plain,2);
            
            logger.message('Computing sample standard deviation.');
            
            kept_deviation_t = std(train_sample_plain,0,2);
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = input_geometry;
            
            obj = obj@transforms.reversible(input_geometry,output_geometry,logger);
            obj.kept_mean = kept_mean_t;
            obj.kept_deviation = kept_deviation_t;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            logger.message('Substracting mean from each instance and dividing by standard deviation.');
            
            sample_coded_t1 = bsxfun(@minus,sample_plain,obj.kept_mean);
            sample_coded = bsxfun(@rdivide,sample_coded_t1,obj.kept_deviation);
        end
        
        function [sample_plain_hat] = do_decode(obj,sample_coded,logger)
            logger.message('Multiplying by standard deviation and adding saved mean to each instance.');
            
            sample_plain_hat_t1 = bsxfun(@times,sample_coded,obj.kept_deviation);
            sample_plain_hat = bsxfun(@plus,sample_plain_hat_t1,obj.kept_mean);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.record.standardize".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0; 0 0.3],10000)';
            
            t = transforms.record.standardize(s,log);
            
            assert(tc.same(t.kept_mean,[3;3],'Epsilon',0.1));
            assert(tc.same(t.kept_deviation,[1;sqrt(0.3)],'Epsilon',0.1));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,2));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing sample mean.\n',...
                                                          'Computing sample standard deviation.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0; 0 0.3],10000)';
            
            t = transforms.record.standardize(s,log);
            s_p = t.code(s,log);
            
            assert(tc.same(s_p,(s - repmat([3;3],1,10000)) ./ repmat([1;sqrt(0.3)],1,10000),'Epsilon',0.1));
            assert(tc.same(mean(s_p,2),[0;0],'Epsilon',0.1));
            assert(tc.same(std(s_p,0,2),[1;1],'Epsilon',0.1));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing sample mean.\n',...
                                                          'Computing sample standard deviation.\n',...
                                                          'Substracting mean from each instance and dividing by standard deviation.\n'))));
                                                      
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
                title('Standardized sample.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0; 0 0.3],10000)';
            
            t = transforms.record.standardize(s,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.same(s_r,s,'Epsilon',0.1));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing sample mean.\n',...
                                                          'Computing sample standard deviation.\n',...
                                                          'Substracting mean from each instance and dividing by standard deviation.\n',...
                                                          'Multiplying by standard deviation and adding saved mean to each instance.\n'))));
                                                      
            if exist('display','var') && (display == true)
                figure();
                subplot(1,3,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original sample.');
                subplot(1,3,2);
                scatter(s_p(1,:),s_p(2,:),'x');
                axis([-4 6 -4 6]);
                axis('square');
                title('Standardized sample.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'.');
                axis([-4 6 -4 6]);
                axis('square');
                title('Restored sample.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  With non-independent features.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0.4; 0.4 0.3],10000)';
            
            t = transforms.record.standardize(s,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.same(s_p,(s - repmat([3;3],1,10000)) ./ repmat([1;sqrt(0.3)],1,10000),'Epsilon',0.1));
            assert(tc.same(mean(s_p,2),[0;0],'Epsilon',0.1));
            assert(tc.same(std(s_p,0,2),[1;1],'Epsilon',0.1));
            assert(tc.same(s_r,s,'Epsilon',0.1));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing sample mean.\n',...
                                                          'Computing sample standard deviation.\n',...
                                                          'Substracting mean from each instance and dividing by standard deviation.\n',...
                                                          'Multiplying by standard deviation and adding saved mean to each instance.\n'))));
                                                      
            if exist('display','var') && (display == true)
                figure();
                subplot(1,3,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original sample.');
                subplot(1,3,2);
                scatter(s_p(1,:),s_p(2,:),'x');
                axis([-4 6 -4 6]);
                axis('square');
                title('Standardized sample.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'.');
                axis([-4 6 -4 6]);
                axis('square');
                title('Restored sample.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
