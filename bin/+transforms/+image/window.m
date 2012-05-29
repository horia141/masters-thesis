classdef window < transform
    properties (GetAccess=public,SetAccess=immutable)
        filter;
        sigma;
    end
    
    methods (Access=public)
        function [obj] = window(train_sample_plain,sigma,logger)
            assert(tc.dataset_image(train_sample_plain));
            assert(size(train_sample_plain,1) == size(train_sample_plain,2)); % A BIT OF A HACK
            assert(tc.scalar(sigma));
            assert(tc.number(sigma));
            assert(sigma > 0);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            [d,dr,dc,dl] = dataset.geometry(train_sample_plain);
            
            filter_t = fspecial('gaussian',dr,sigma);
            filter_t = filter_t ./ max(max(filter_t));
            
            input_geometry = [d,dr,dc,dl];
            output_geometry = input_geometry;
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.filter = filter_t;
            obj.sigma = sigma;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            N = dataset.count(sample_plain);
            [~,~,~,dl] = dataset.geometry(sample_plain);
            
            logger.message('Applying filter.');
            
            bulk_filter = repmat(obj.filter,[1 1 dl N]);
            sample_coded = sample_plain .* bulk_filter;
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.window".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = dataset.load_image_from_dir('../test/scenes_small','gray',[192 192]);
            
            t = transforms.image.window(s,10,log);
            
            assert(tc.matrix(t.filter));
            assert(tc.same(size(t.filter),[192 192]));
            assert(tc.unitreal(abs(t.filter)));
            assert(max(max(t.filter)) == 1);
            assert(t.sigma == 10);
            assert(tc.same(t.input_geometry,[192*192*1 192 192 1]));
            assert(tc.same(t.output_geometry,[192*192*1 192 192 1]));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = dataset.load_image_from_dir('../test/scenes_small','gray',[192 192]);
            
            t = transforms.image.window(s,60,log);
            s_p = t.code(s,log);
            
            assert(tc.tensor(s_p,4));
            assert(tc.same(size(s_p),[192 192 1 7]));
            assert(tc.unitreal(s_p));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s,4,2));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p,4,2));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = dataset.load_image_from_dir('../test/scenes_small','original',[192 192]);
            
            t = transforms.image.window(s,60,log);
            s_p = t.code(s,log);
            
            assert(tc.tensor(s_p,4));
            assert(tc.same(size(s_p),[192 192 3 7]));
            assert(tc.unitreal(s_p));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s,4,2));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p,4,2));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
