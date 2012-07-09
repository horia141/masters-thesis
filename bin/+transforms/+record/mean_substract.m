classdef mean_substract < transform
    properties (GetAccess=public,SetAccess=immutable)
        kept_mean;
    end

    methods (Access=public)
        function [obj] = mean_substract(train_sample_plain,logger)
            assert(check.dataset_record(train_sample_plain));
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
            assert(logger.active);

            logger.message('Computing dataset mean.');

            kept_mean_t = mean(train_sample_plain,2);

            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = input_geometry;

            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.kept_mean = kept_mean_t;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            logger.message('Substracting mean from each sample.');

            sample_coded = bsxfun(@minus,sample_plain,obj.kept_mean);
        end
    end
        
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.mean_substract".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.correlated_cloud();

            t = transforms.record.mean_substract(s,logg);
            
            assert(check.same(t.kept_mean,[3;3],0.1));
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,2));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.correlated_cloud();
            
            t = transforms.record.mean_substract(s,logg);
            s_p = t.code(s,logg);
            
            assert(check.same(s_p,s - repmat([3;3],1,10000),0.1));
            assert(check.same(mean(s_p,2),[0;0],0.1));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p(1,:),s_p(2,:),'x');
                axis([-4 6 -4 6]);
                axis('square');
                title('Mean substracted samples.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
        end
    end
end
