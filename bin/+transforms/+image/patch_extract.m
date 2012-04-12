classdef patch_extract < transform
    properties (GetAccess=public,SetAccess=immutable)
        patches_count;
        patch_row_count;
        patch_col_count;
        required_variance;
    end
        
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = patch_extract(train_image_plain,patches_count,patch_row_count,patch_col_count,required_variance,logger)
            assert(tc.scalar(train_image_plain));
            assert(tc.datasets_image(train_image_plain));
            assert(train_image_plain.samples_count >= 1);
            assert(tc.scalar(patches_count));
            assert(tc.natural(patches_count));
            assert(patches_count >= 1);
            assert(tc.scalar(patch_row_count));
            assert(tc.natural(patch_row_count));
            assert(patch_row_count >= 1);
            assert(tc.scalar(patch_col_count));
            assert(tc.natural(patch_col_count));
            assert(patch_col_count >= 1);
            assert(tc.scalar(required_variance));
            assert(tc.number(required_variance));
            assert(required_variance >= 0);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            obj = obj@transform(logger);
            obj.patches_count = 1;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.required_variance = required_variance;
            
            logger.beg_node('Extracting plain/coded samples');
            
            obj.one_sample_plain = train_image_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain,logger);
            obj.patches_count = patches_count; % HACK DI HACK HACK %
            
            logger.end_node();
        end
    end
    
    methods (Access=protected)
        function [image_coded] = do_code(obj,image_plain,logger)
            log_batch_size = ceil(obj.patches_count / 10);
            images_coded = zeros(obj.patch_row_count,obj.patch_col_count,image_plain.layers_count,obj.patches_count);
            
            curr_patches_count = 1;
            
            logger.beg_node('Extracting patches');
            
            while curr_patches_count <= obj.patches_count
                image_idx = randi(image_plain.samples_count);
                row_skip = randi(image_plain.row_count - obj.patch_row_count + 1);
                col_skip = randi(image_plain.col_count - obj.patch_col_count + 1);
                
                patch = image_plain.images(row_skip:(row_skip + obj.patch_row_count - 1),...
                                           col_skip:(col_skip + obj.patch_col_count - 1),...
                                           :,image_idx);
                
                if var(patch(:)) > obj.required_variance
                    if mod(curr_patches_count-1,log_batch_size) == 0
                        logger.message('Patches %d to %d.',curr_patches_count,min(curr_patches_count+log_batch_size-1,obj.patches_count));
                    end

                    images_coded(:,:,:,curr_patches_count) = patch;
                    curr_patches_count = curr_patches_count + 1;
                end
            end
            
            logger.end_node();

            logger.message('Building dataset.');
            
            image_coded = datasets.image({'none'},images_coded,ones(obj.patches_count,1));
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.patch_extract".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.patch_extract(s,10,5,5,0.01,log);
            
            assert(t.patches_count == 10);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.required_variance == 0.01);
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
            assert(tc.check(size(t.one_sample_coded.samples) == [1 25]));
            assert(tc.matrix(t.one_sample_coded.samples) && tc.unitreal(t.one_sample_coded.samples));
            assert(tc.check(t.one_sample_coded.labels_idx == s.labels_idx(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 5*5);
            assert(tc.check(size(t.one_sample_coded.images) == [5 5]));
            assert(tc.tensor(t.one_sample_coded.images,4) && tc.unitreal(t.one_sample_coded.images));
            assert(var(t.one_sample_coded.images(:)) >= 0.01);
            assert(t.one_sample_coded.layers_count == 1);
            assert(t.one_sample_coded.row_count == 5);
            assert(t.one_sample_coded.col_count == 5);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Extracting patches:\n',...
                                                          '    Patches 1 to 1.\n',...
                                                          '  Building dataset.\n'))));
                                                      
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.patch_extract(s,50,40,40,0.01,log);
            s_p = t.code(s,log);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(tc.check(size(s_p.samples) == [50 40*40]));
            assert(tc.matrix(s_p.samples) && tc.unitreal(s_p.samples));
            assert(tc.check(s_p.labels_idx == ones(50,1)));
            assert(s_p.samples_count == 50);
            assert(s_p.features_count == 40*40);
            assert(tc.check(size(s_p.images) == [40 40 1 50]));
            assert(tc.tensor(s_p.images,4) && tc.unitreal(s_p.images));
            assert(tc.check(arrayfun(@(ii)var(reshape(s_p.images(:,:,:,ii),[1 40*40])) >= 0.01,1:50)));
            assert(s_p.layers_count == 1);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Extracting patches:\n',...
                                                          '    Patches 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Extracting patches:\n',...
                                                          '  Patches 1 to 5.\n',...
                                                          '  Patches 6 to 10.\n',...
                                                          '  Patches 11 to 15.\n',...
                                                          '  Patches 16 to 20.\n',...
                                                          '  Patches 21 to 25.\n',...
                                                          '  Patches 26 to 30.\n',...
                                                          '  Patches 31 to 35.\n',...
                                                          '  Patches 36 to 40.\n',...
                                                          '  Patches 41 to 45.\n',...
                                                          '  Patches 46 to 50.\n',...
                                                          'Building dataset.\n'))));
                                                          

            if exist('display','var') && (display == true)
                figure();
                imshow(utils.format_as_tiles(s_p.images,5,10));
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
            
            t = transforms.image.patch_extract(s,50,40,40,0.01,log);
            s_p = t.code(s,log);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(tc.check(size(s_p.samples) == [50 3*40*40]));
            assert(tc.matrix(s_p.samples) && tc.unitreal(s_p.samples));
            assert(tc.check(s_p.labels_idx == ones(50,1)));
            assert(s_p.samples_count == 50);
            assert(s_p.features_count == 3*40*40);
            assert(tc.check(size(s_p.images) == [40 40 3 50]));
            assert(tc.tensor(s_p.images,4) && tc.unitreal(s_p.images));
            assert(tc.check(arrayfun(@(ii)var(reshape(s_p.images(:,:,:,ii),[1 3*40*40])) >= 0.01,1:50)));
            assert(s_p.layers_count == 3);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Extracting patches:\n',...
                                                          '    Patches 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Extracting patches:\n',...
                                                          '  Patches 1 to 5.\n',...
                                                          '  Patches 6 to 10.\n',...
                                                          '  Patches 11 to 15.\n',...
                                                          '  Patches 16 to 20.\n',...
                                                          '  Patches 21 to 25.\n',...
                                                          '  Patches 26 to 30.\n',...
                                                          '  Patches 31 to 35.\n',...
                                                          '  Patches 36 to 40.\n',...
                                                          '  Patches 41 to 45.\n',...
                                                          '  Patches 46 to 50.\n',...
                                                          'Building dataset.\n'))));

            if exist('display','var') && (display == true)
                figure();
                imshow(utils.format_as_tiles(s_p.images,5,10));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
