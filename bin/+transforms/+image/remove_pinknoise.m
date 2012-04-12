classdef remove_pinknoise < transform        
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = remove_pinknoise(train_image_plain,logger)
            assert(tc.scalar(train_image_plain));
            assert(tc.datasets_image(train_image_plain));
            assert(train_image_plain.samples_count >= 1);
            assert(train_image_plain.row_count == train_image_plain.col_count);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            obj = obj@transform(logger);
            
            logger.beg_node('Extracting plain/coded samples');
            
            obj.one_sample_plain = train_image_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain,logger);
            
            logger.end_node();
        end
    end
    
    methods (Access=protected)
        function [image_coded] = do_code(~,image_plain,logger)
            N = image_plain.row_count;
            
            logger.message('Building filter kernel.');
            
            [fx fy] = meshgrid((-N/2):(N/2-1),(-N/2):(N/2 - 1));
            rho = sqrt(fx .* fx + fy .* fy);
            f_0 = 0.4 * N;
            filter_kernel = fftshift(rho .* exp(-(rho/f_0) .^ 4));
            
            log_batch_size = ceil(image_plain.samples_count / 10);
            images_coded = zeros(image_plain.row_count,image_plain.col_count,image_plain.layers_count,image_plain.samples_count);
            
            logger.beg_node('Building filtered images');
            
            for layer = 1:image_plain.layers_count
                logger.beg_node('Layer %d',layer);
                
                for ii = 1:image_plain.samples_count
                    if mod(ii-1,log_batch_size) == 0
                        logger.message('Processing images %d to %d.',ii,min(ii+log_batch_size-1,image_plain.samples_count));
                    end

                    imagew = real(ifft2(fft2(image_plain.images(:,:,layer,ii)) .* filter_kernel));
                    images_coded(:,:,layer,ii) = imagew;
                end
                
                logger.end_node();
            end
            
            logger.end_node();
            
            logger.message('Building dataset.');
            
            image_coded = datasets.image(image_plain.classes,images_coded,image_plain.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.remove_pinknoise".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../test/scenes_small','gray',[192 192]);
            
            t = transforms.image.remove_pinknoise(s,log);
            
            assert(length(t.one_sample_plain.classes) == 1);
            assert(strcmp(t.one_sample_plain.classes{1},'none'));
            assert(t.one_sample_plain.classes_count == 1);
            assert(tc.check(t.one_sample_plain.samples == s.samples(1,:)));
            assert(tc.check(t.one_sample_plain.labels_idx == s.labels_idx(1)));
            assert(t.one_sample_plain.samples_count == 1);
            assert(t.one_sample_plain.features_count == 192*192);
            assert(tc.check(t.one_sample_plain.images == s.images(:,:,:,1)));
            assert(t.one_sample_plain.layers_count == 1);
            assert(t.one_sample_plain.row_count == 192);
            assert(t.one_sample_plain.col_count == 192);
            assert(t.one_sample_plain.compatible(s));
            assert(length(t.one_sample_coded.classes) == 1);
            assert(strcmp(t.one_sample_coded.classes{1},'none'));
            assert(t.one_sample_coded.classes_count == 1);
            assert(tc.check(size(t.one_sample_coded.samples) == [1 192*192]));
            assert(tc.matrix(t.one_sample_coded.samples) && tc.number(t.one_sample_coded.samples));
            assert(tc.check(t.one_sample_coded.labels_idx == s.labels_idx(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 192*192);
            assert(tc.check(size(t.one_sample_coded.images) == [192 192]));
            assert(tc.tensor(t.one_sample_coded.images,4) && tc.number(t.one_sample_coded.images));
            assert(t.one_sample_coded.layers_count == 1);
            assert(t.one_sample_coded.row_count == 192);
            assert(t.one_sample_coded.col_count == 192);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Building filter kernel.\n',...
                                                          '  Building filtered images:\n',...
                                                          '    Layer 1:\n',...
                                                          '      Processing images 1 to 1.\n',...
                                                          '  Building dataset.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../test/scenes_small');
            p_t = transforms.image.patch_extract(s,80,40,40,0.0001,log);
            p_s = p_t.code(s,log);
            
            t = transforms.image.remove_pinknoise(p_s,log);
            s_p = t.code(p_s,log);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(tc.check(size(s_p.samples) == [80 1600]));
            assert(tc.matrix(s_p.samples) && tc.number(s_p.samples));
            assert(tc.check(s_p.labels_idx == ones(80,1)));
            assert(s_p.samples_count == 80);
            assert(s_p.features_count == 1600);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            assert(tc.check(size(s_p.images) == [40 40 1 80]));
            assert(tc.tensor(s_p.images,4) && tc.number(s_p.images));
            assert(s_p.layers_count == 1);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Extracting patches:\n',...
                                                          '    Patches 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Extracting patches:\n',...
                                                          '  Patches 1 to 8.\n',...
                                                          '  Patches 9 to 16.\n',...
                                                          '  Patches 17 to 24.\n',...
                                                          '  Patches 25 to 32.\n',...
                                                          '  Patches 33 to 40.\n',...
                                                          '  Patches 41 to 48.\n',...
                                                          '  Patches 49 to 56.\n',...
                                                          '  Patches 57 to 64.\n',...
                                                          '  Patches 65 to 72.\n',...
                                                          '  Patches 73 to 80.\n',...
                                                          'Building dataset.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Building filter kernel.\n',...
                                                          '  Building filtered images:\n',...
                                                          '    Layer 1:\n',...
                                                          '      Processing images 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building filter kernel.\n',...
                                                          'Building filtered images:\n',...
                                                          '  Layer 1:\n',...
                                                          '    Processing images 1 to 8.\n',...
                                                          '    Processing images 9 to 16.\n',...
                                                          '    Processing images 17 to 24.\n',...
                                                          '    Processing images 25 to 32.\n',...
                                                          '    Processing images 33 to 40.\n',...
                                                          '    Processing images 41 to 48.\n',...
                                                          '    Processing images 49 to 56.\n',...
                                                          '    Processing images 57 to 64.\n',...
                                                          '    Processing images 65 to 72.\n',...
                                                          '    Processing images 73 to 80.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(p_s.images,8,10));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p.images,'global'),8,10));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../test/scenes_small','color');
            p_t = transforms.image.patch_extract(s,80,40,40,0.0001,log);
            p_s = p_t.code(s,log);
            
            t = transforms.image.remove_pinknoise(p_s,log);
            s_p = t.code(p_s,log);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(tc.check(size(s_p.samples) == [80 3*40*40]));
            assert(tc.matrix(s_p.samples) && tc.number(s_p.samples));
            assert(tc.check(s_p.labels_idx == ones(80,1)));
            assert(s_p.samples_count == 80);
            assert(s_p.features_count == 3*40*40);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            assert(tc.check(size(s_p.images) == [40 40 3 80]));
            assert(tc.tensor(s_p.images,4) && tc.number(s_p.images));
            assert(s_p.layers_count == 3);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Extracting patches:\n',...
                                                          '    Patches 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Extracting patches:\n',...
                                                          '  Patches 1 to 8.\n',...
                                                          '  Patches 9 to 16.\n',...
                                                          '  Patches 17 to 24.\n',...
                                                          '  Patches 25 to 32.\n',...
                                                          '  Patches 33 to 40.\n',...
                                                          '  Patches 41 to 48.\n',...
                                                          '  Patches 49 to 56.\n',...
                                                          '  Patches 57 to 64.\n',...
                                                          '  Patches 65 to 72.\n',...
                                                          '  Patches 73 to 80.\n',...
                                                          'Building dataset.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Building filter kernel.\n',...
                                                          '  Building filtered images:\n',...
                                                          '    Layer 1:\n',...
                                                          '      Processing images 1 to 1.\n',...
                                                          '    Layer 2:\n',...
                                                          '      Processing images 1 to 1.\n',...
                                                          '    Layer 3:\n',...
                                                          '      Processing images 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building filter kernel.\n',...
                                                          'Building filtered images:\n',...
                                                          '  Layer 1:\n',...
                                                          '    Processing images 1 to 8.\n',...
                                                          '    Processing images 9 to 16.\n',...
                                                          '    Processing images 17 to 24.\n',...
                                                          '    Processing images 25 to 32.\n',...
                                                          '    Processing images 33 to 40.\n',...
                                                          '    Processing images 41 to 48.\n',...
                                                          '    Processing images 49 to 56.\n',...
                                                          '    Processing images 57 to 64.\n',...
                                                          '    Processing images 65 to 72.\n',...
                                                          '    Processing images 73 to 80.\n',...
                                                          '  Layer 2:\n',...
                                                          '    Processing images 1 to 8.\n',...
                                                          '    Processing images 9 to 16.\n',...
                                                          '    Processing images 17 to 24.\n',...
                                                          '    Processing images 25 to 32.\n',...
                                                          '    Processing images 33 to 40.\n',...
                                                          '    Processing images 41 to 48.\n',...
                                                          '    Processing images 49 to 56.\n',...
                                                          '    Processing images 57 to 64.\n',...
                                                          '    Processing images 65 to 72.\n',...
                                                          '    Processing images 73 to 80.\n',...
                                                          '  Layer 3:\n',...
                                                          '    Processing images 1 to 8.\n',...
                                                          '    Processing images 9 to 16.\n',...
                                                          '    Processing images 17 to 24.\n',...
                                                          '    Processing images 25 to 32.\n',...
                                                          '    Processing images 33 to 40.\n',...
                                                          '    Processing images 41 to 48.\n',...
                                                          '    Processing images 49 to 56.\n',...
                                                          '    Processing images 57 to 64.\n',...
                                                          '    Processing images 65 to 72.\n',...
                                                          '    Processing images 73 to 80.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(p_s.images,8,10));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p.images,'global'),8,10));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
