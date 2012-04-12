classdef window < transform
    properties (GetAccess=public,SetAccess=immutable)
        filter;
        sigma;
    end
        
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = window(train_image_plain,sigma,logger)
            assert(tc.scalar(train_image_plain));
            assert(tc.datasets_image(train_image_plain));
            assert(train_image_plain.samples_count >= 1);
            assert(train_image_plain.row_count == train_image_plain.col_count);
            assert(tc.scalar(sigma));
            assert(tc.number(sigma));
            assert(sigma > 0);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            filter_t = fspecial('gaussian',train_image_plain.row_count,sigma);
            filter_t = filter_t ./ max(max(filter_t));
            
            obj = obj@transform(logger);
            obj.filter = filter_t;
            obj.sigma = sigma;
            
            logger.beg_node('Extracting plain/coded samples');
            
            obj.one_sample_plain = train_image_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain,logger);
            
            logger.end_node();
        end
    end
    
    methods (Access=protected)
        function [image_coded] = do_code(obj,image_plain,logger)
            log_batch_size = ceil(image_plain.samples_count / 10);
            images_coded = zeros(image_plain.row_count,image_plain.col_count,image_plain.layers_count,image_plain.samples_count);
            
            logger.beg_node('Building windowed images');
            
            for layer = 1:image_plain.layers_count
                logger.beg_node('Layer %d',layer);

                for ii = 1:image_plain.samples_count
                    if mod(ii-1,log_batch_size) == 0
                        logger.message('Processing images %d to %d.',ii,min(ii+log_batch_size-1,image_plain.samples_count));
                    end

                    images_coded(:,:,layer,ii) = image_plain.images(:,:,layer,ii) .* obj.filter;
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
            fprintf('Testing "transforms.image.window".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../test/scenes_small','gray',[192 192]);
            
            t = transforms.image.window(s,10,log);
            
            assert(tc.check(size(t.filter) == [192 192]));
            assert(max(max(t.filter)) == 1);
            assert(t.sigma == 10);
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
            assert(tc.matrix(t.one_sample_coded.samples) && tc.unitreal(t.one_sample_coded.samples));
            assert(tc.check(t.one_sample_coded.labels_idx == s.labels_idx(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 192*192);
            assert(tc.check(size(t.one_sample_coded.images) == [192 192]));
            assert(tc.tensor(t.one_sample_coded.images,4) && tc.unitreal(t.one_sample_coded.images));
            assert(t.one_sample_coded.layers_count == 1);
            assert(t.one_sample_coded.row_count == 192);
            assert(t.one_sample_coded.col_count == 192);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Building windowed images:\n',...
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
            s = datasets.image.load_from_dir('../test/scenes_small','gray',[192 192]);
            
            t = transforms.image.window(s,60,log);
            s_p = t.code(s,log);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(tc.check(size(s_p.samples) == [7 192*192]));
            assert(tc.matrix(s_p.samples) && tc.unitreal(s_p.samples));
            assert(tc.check(s_p.labels_idx == s.labels_idx));
            assert(s_p.samples_count == 7);
            assert(s_p.features_count == 192*192);
            assert(s_p.row_count == 192);
            assert(s_p.col_count == 192);
            assert(tc.check(size(s_p.images) == [192 192 1 7]));
            assert(tc.tensor(s_p.images,4) && tc.unitreal(s_p.images));
            assert(s_p.layers_count == 1);
            assert(s_p.row_count == 192);
            assert(s_p.col_count == 192);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Building windowed images:\n',...
                                                          '    Layer 1:\n',...
                                                          '      Processing images 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building windowed images:\n',...
                                                          '  Layer 1:\n',...
                                                          '    Processing images 1 to 1.\n',...
                                                          '    Processing images 2 to 2.\n',...
                                                          '    Processing images 3 to 3.\n',...
                                                          '    Processing images 4 to 4.\n',...
                                                          '    Processing images 5 to 5.\n',...
                                                          '    Processing images 6 to 6.\n',...
                                                          '    Processing images 7 to 7.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s.images,4,2));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p.images,4,2));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../test/scenes_small','color',[192 192]);
            
            t = transforms.image.window(s,60,log);
            s_p = t.code(s,log);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(tc.check(size(s_p.samples) == [7 3*192*192]));
            assert(tc.matrix(s_p.samples) && tc.unitreal(s_p.samples));
            assert(tc.check(s_p.labels_idx == s.labels_idx));
            assert(s_p.samples_count == 7);
            assert(s_p.features_count == 3*192*192);
            assert(s_p.row_count == 192);
            assert(s_p.col_count == 192);
            assert(tc.check(size(s_p.images) == [192 192 3 7]));
            assert(tc.tensor(s_p.images,4) && tc.unitreal(s_p.images));
            assert(s_p.layers_count == 3);
            assert(s_p.row_count == 192);
            assert(s_p.col_count == 192);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Building windowed images:\n',...
                                                          '    Layer 1:\n',...
                                                          '      Processing images 1 to 1.\n',...
                                                          '    Layer 2:\n',...
                                                          '      Processing images 1 to 1.\n',...
                                                          '    Layer 3:\n',...
                                                          '      Processing images 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building windowed images:\n',...
                                                          '  Layer 1:\n',...
                                                          '    Processing images 1 to 1.\n',...
                                                          '    Processing images 2 to 2.\n',...
                                                          '    Processing images 3 to 3.\n',...
                                                          '    Processing images 4 to 4.\n',...
                                                          '    Processing images 5 to 5.\n',...
                                                          '    Processing images 6 to 6.\n',...
                                                          '    Processing images 7 to 7.\n',...
                                                          '  Layer 2:\n',...
                                                          '    Processing images 1 to 1.\n',...
                                                          '    Processing images 2 to 2.\n',...
                                                          '    Processing images 3 to 3.\n',...
                                                          '    Processing images 4 to 4.\n',...
                                                          '    Processing images 5 to 5.\n',...
                                                          '    Processing images 6 to 6.\n',...
                                                          '    Processing images 7 to 7.\n',...
                                                          '  Layer 3:\n',...
                                                          '    Processing images 1 to 1.\n',...
                                                          '    Processing images 2 to 2.\n',...
                                                          '    Processing images 3 to 3.\n',...
                                                          '    Processing images 4 to 4.\n',...
                                                          '    Processing images 5 to 5.\n',...
                                                          '    Processing images 6 to 6.\n',...
                                                          '    Processing images 7 to 7.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s.images,4,2));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p.images,4,2));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end