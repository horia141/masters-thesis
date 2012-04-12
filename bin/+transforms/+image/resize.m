classdef resize < transform
    properties (GetAccess=public,SetAccess=immutable)
        new_row_count;
        new_col_count;
    end
        
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = resize(train_image_plain,new_row_count,new_col_count,logger)
            assert(tc.scalar(train_image_plain));
            assert(tc.datasets_image(train_image_plain));
            assert(train_image_plain.samples_count >= 1);
            assert(tc.scalar(new_row_count));
            assert(tc.natural(new_row_count));
            assert(new_row_count >= 1);
            assert(tc.scalar(new_col_count));
            assert(tc.natural(new_col_count));
            assert(new_col_count >= 1);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            obj = obj@transform(logger);
            obj.new_row_count = new_row_count;
            obj.new_col_count = new_col_count;

            logger.beg_node('Extracting plain/coded samples');
            
            obj.one_sample_plain = train_image_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain,logger);
            
            logger.end_node();
        end
    end
    
    methods (Access=protected)
        function [image_coded] = do_code(obj,image_plain,logger)
            if mod(image_plain.row_count,obj.new_row_count) == 0 && ...
               mod(image_plain.col_count,obj.new_col_count) == 0
                logger.message('Building resized images (fast because new size is a sub-multiple of old size).');

                row_step = image_plain.row_count / obj.new_row_count;
                col_step = image_plain.col_count / obj.new_col_count;

                images_coded = image_plain.images(1:row_step:end,1:col_step:end,:,:);

                logger.message('Building dataset.');

                image_coded = datasets.image(image_plain.classes,images_coded,image_plain.labels_idx);
            else
                log_batch_size = ceil(image_plain.samples_count / 10);
                images_coded = zeros(obj.new_row_count,obj.new_col_count,image_plain.layers_count,image_plain.samples_count);
            
                logger.beg_node('Building resized images');
            
                for ii = 1:image_plain.samples_count
                    if mod(ii-1,log_batch_size) == 0
                        logger.message('Processing images %d to %d.',ii,min(ii+log_batch_size-1,image_plain.samples_count));
                    end
                
                    images_coded(:,:,:,ii) = imresize(image_plain.images(:,:,:,ii),[obj.new_row_count obj.new_col_count]);
                end
            
                logger.end_node();
            
                logger.message('Building dataset.');
            
                images_coded = utils.clamp_images_to_unit(images_coded);
                image_coded = datasets.image(image_plain.classes,images_coded,image_plain.labels_idx);
            end
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.resize".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.resize(s,20,20,log);
            
            assert(t.new_row_count == 20);
            assert(t.new_col_count == 20);
            assert(length(t.one_sample_plain.classes) == 1);
            assert(strcmp(t.one_sample_plain.classes{1},'none'));
            assert(t.one_sample_plain.classes_count == 1);
            assert(tc.check(t.one_sample_plain.samples == s.samples(1,:)));
            assert(tc.check(t.one_sample_plain.labels_idx == s.labels_idx(1)));
            assert(t.one_sample_plain.samples_count == 1);
            assert(t.one_sample_plain.features_count == 192*256);
            assert(tc.check(t.one_sample_plain.images == s.images(:,:,:,1)));
            assert(t.one_sample_plain.layers_count == 1);
            assert(t.one_sample_plain.row_count == 192);
            assert(t.one_sample_plain.col_count == 256);
            assert(t.one_sample_plain.compatible(s));
            assert(length(t.one_sample_coded.classes) == 1);
            assert(strcmp(t.one_sample_coded.classes{1},'none'));
            assert(t.one_sample_coded.classes_count == 1);
            assert(tc.check(size(t.one_sample_coded.samples) == [1 20*20]));
            assert(tc.matrix(t.one_sample_coded.samples) && tc.unitreal(t.one_sample_coded.samples));
            assert(tc.check(t.one_sample_coded.labels_idx == s.labels_idx(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 20*20);
            assert(tc.check(size(t.one_sample_coded.images) == [20 20]));
            assert(tc.tensor(t.one_sample_coded.images,4) && tc.unitreal(t.one_sample_coded.images));
            assert(t.one_sample_coded.layers_count == 1);
            assert(t.one_sample_coded.row_count == 20);
            assert(t.one_sample_coded.col_count == 20);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Building resized images:\n',...
                                                          '    Processing images 1 to 1.\n',...
                                                          '  Building dataset.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.resize(s,100,100,log);
            s_p = t.code(s,log);
            
            assert(tc.same(s_p.classes,s.classes));
            assert(s_p.classes_count == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(tc.check(arrayfun(@(ii)tc.same(s_p.samples(ii,:),reshape(utils.clamp_images_to_unit(imresize(s.images(:,:,:,ii),[100 100])),[1 100*100])),1:7)));
            assert(tc.check(s_p.labels_idx == s.labels_idx));
            assert(s_p.samples_count == 7);
            assert(s_p.features_count == 100*100);
            assert(tc.check(arrayfun(@(ii)tc.same(s_p.images(:,:,:,ii),utils.clamp_images_to_unit(imresize(s.images(:,:,:,ii),[100 100]))),1:7)));
            assert(s_p.layers_count == 1);
            assert(s_p.row_count == 100);
            assert(s_p.col_count == 100);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Building resized images:\n',...
                                                          '    Processing images 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building resized images:\n',...
                                                          '  Processing images 1 to 1.\n',...
                                                          '  Processing images 2 to 2.\n',...
                                                          '  Processing images 3 to 3.\n',...
                                                          '  Processing images 4 to 4.\n',...
                                                          '  Processing images 5 to 5.\n',...
                                                          '  Processing images 6 to 6.\n',...
                                                          '  Processing images 7 to 7.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s.images,3,3));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p.images,3,3));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../data/test/scenes_small','color');
            
            t = transforms.image.resize(s,100,100,log);
            s_p = t.code(s,log);
            
            assert(tc.same(s_p.classes,s.classes));
            assert(s_p.classes_count == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(tc.check(arrayfun(@(ii)tc.same(s_p.samples(ii,:),reshape(utils.clamp_images_to_unit(imresize(s.images(:,:,:,ii),[100 100])),[1 3*100*100])),1:7)));
            assert(tc.check(s_p.labels_idx == s.labels_idx));
            assert(s_p.samples_count == 7);
            assert(s_p.features_count == 3*100*100);
            assert(tc.check(arrayfun(@(ii)tc.same(s_p.images(:,:,:,ii),utils.clamp_images_to_unit(imresize(s.images(:,:,:,ii),[100 100]))),1:7)));
            assert(s_p.layers_count == 3);
            assert(s_p.row_count == 100);
            assert(s_p.col_count == 100);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Building resized images:\n',...
                                                          '    Processing images 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building resized images:\n',...
                                                          '  Processing images 1 to 1.\n',...
                                                          '  Processing images 2 to 2.\n',...
                                                          '  Processing images 3 to 3.\n',...
                                                          '  Processing images 4 to 4.\n',...
                                                          '  Processing images 5 to 5.\n',...
                                                          '  Processing images 6 to 6.\n',...
                                                          '  Processing images 7 to 7.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s.images,3,3));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p.images,3,3));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With grayscale images and sub-multiple size.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.resize(s,96,128,log);
            s_p = t.code(s,log);
            
            assert(tc.same(s_p.classes,s.classes));
            assert(s_p.classes_count == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(tc.check(arrayfun(@(ii)tc.same(s_p.samples(ii,:),reshape(s.images(1:2:end,1:2:end,:,ii),[1 96*128])),1:7)));
            assert(tc.check(s_p.labels_idx == s.labels_idx));
            assert(s_p.samples_count == 7);
            assert(s_p.features_count == 96*128);
            assert(tc.check(arrayfun(@(ii)tc.same(s_p.images(:,:,:,ii),s.images(1:2:end,1:2:end,:,ii)),1:7)));
            assert(s_p.layers_count == 1);
            assert(s_p.row_count == 96);
            assert(s_p.col_count == 128);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Building resized images (fast because new size is a sub-multiple of old size).\n',...
                                                          '  Building dataset.\n',...
                                                          'Building resized images (fast because new size is a sub-multiple of old size).\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s.images,3,3));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p.images,3,3));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
