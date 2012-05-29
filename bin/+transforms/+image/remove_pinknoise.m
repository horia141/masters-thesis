classdef remove_pinknoise < transform
    methods (Access=public)
        function [obj] = remove_pinknoise(train_sample_plain,logger)
            assert(tc.dataset_image(train_sample_plain));
            assert(size(train_sample_plain,1) == size(train_sample_plain,2)); % A BIT OF A HACK
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
            N = dataset.count(sample_plain);
            [~,dr,dc,dl] = dataset.geometry(sample_plain);
            
            logger.message('Building filter kernel.');
            
            [fx fy] = meshgrid((-dr/2):(dr/2-1),(-dr/2):(dr/2 - 1));
            rho = sqrt(fx .* fx + fy .* fy);
            f_0 = 0.4 * dr;
            filter_kernel = fftshift(rho .* exp(-(rho/f_0) .^ 4));
            
            log_batch_size = ceil(N / 10);
            sample_coded = zeros(dr,dc,dl,N);
            
            logger.beg_node('Building filtered images');
            
            for layer = 1:dl
                logger.beg_node('Layer %d',layer);
                
                for ii = 1:N
                    if mod(ii - 1,log_batch_size) == 0
                        logger.message('Processing images %d to %d.',ii,min(ii + log_batch_size - 1,N));
                    end

                    sample_coded(:,:,layer,ii) = real(ifft2(fft2(sample_plain(:,:,layer,ii)) .* filter_kernel));
                end
                
                logger.end_node();
            end
            
            logger.end_node();
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.remove_pinknoise".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = dataset.load_image_from_dir('../test/scenes_small','gray',[192 192]);
            
            t = transforms.image.remove_pinknoise(s,log);
            
            assert(tc.same(t.input_geometry,[192*192*1 192 192 1]));
            assert(tc.same(t.output_geometry,[192*192*1 192 192 1]));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = dataset.load_image_from_dir('../test/scenes_small');
            p_t = transforms.image.patch_extract(s,80,40,40,0.0001,log);
            p_s = p_t.code(s,log);
            
            t = transforms.image.remove_pinknoise(p_s,log);
            s_p = t.code(p_s,log);
            
            assert(tc.tensor(s_p,4));
            assert(tc.same(size(s_p),[40 40 1 80]));
            assert(tc.number(s_p));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(p_s,8,10));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p,'global'),8,10));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = dataset.load_image_from_dir('../test/scenes_small','original');
            p_t = transforms.image.patch_extract(s,80,40,40,0.0001,log);
            p_s = p_t.code(s,log);
            
            t = transforms.image.remove_pinknoise(p_s,log);
            s_p = t.code(p_s,log);
            
            assert(tc.tensor(s_p,4));
            assert(tc.same(size(s_p),[40 40 3 80]));
            assert(tc.number(s_p));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(p_s,8,10));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p,'global'),8,10));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
