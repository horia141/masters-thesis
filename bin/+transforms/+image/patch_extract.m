classdef patch_extract < transform
    properties (GetAccess=public,SetAccess=immutable)
        patches_count;
        patch_row_count;
        patch_col_count;
        required_variance;
    end
    
    methods (Access=public)
        function [obj] = patch_extract(train_sample_plain,patches_count,patch_row_count,patch_col_count,required_variance,logger)
            assert(tc.dataset_image(train_sample_plain));
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
            
            [d dr dc dl] = dataset.geometry(train_sample_plain);
            
            input_geometry = [d dr dc dl];
            output_geometry = [patch_row_count*patch_col_count*dl patch_row_count patch_col_count dl];
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.patches_count = patches_count;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.required_variance = required_variance;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            N = dataset.count(sample_plain);
            [~,dr,dc,dl] = dataset.geometry(sample_plain);
            
            log_batch_size = ceil(obj.patches_count / 10);
            sample_coded = zeros(obj.patch_row_count,obj.patch_col_count,dl,N);
            
            curr_patches_count = 1;
            
            logger.beg_node('Extracting patches');
            
            while curr_patches_count <= obj.patches_count
                image_idx = randi(N);
                row_skip = randi(dr - obj.patch_row_count + 1);
                col_skip = randi(dc - obj.patch_col_count + 1);
                
                patch = sample_plain(row_skip:(row_skip + obj.patch_row_count - 1),...
                                     col_skip:(col_skip + obj.patch_col_count - 1),...
                                     :,image_idx);
                
                if var(patch(:)) > obj.required_variance
                    if mod(curr_patches_count - 1,log_batch_size) == 0
                        logger.message('Patches %d to %d.',curr_patches_count,min(curr_patches_count + log_batch_size - 1,obj.patches_count));
                    end

                    sample_coded(:,:,:,curr_patches_count) = patch;
                    curr_patches_count = curr_patches_count + 1;
                end
            end
            
            logger.end_node();
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.patch_extract".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = dataset.load_image_from_dir('../test/scenes_small');
            
            t = transforms.image.patch_extract(s,10,5,5,0.01,log);
            
            assert(t.patches_count == 10);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.required_variance == 0.01);
            assert(tc.same(t.input_geometry,[192*256*1 192 256 1]));
            assert(tc.same(t.output_geometry,[5*5*1 5 5 1]));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = dataset.load_image_from_dir('../test/scenes_small');
            
            t = transforms.image.patch_extract(s,50,40,40,0.01,log);
            s_p = t.code(s,log);
            
            assert(tc.tensor(s_p,4));
            assert(tc.same(size(s_p),[40 40 1 50]));
            assert(tc.unitreal(s_p));
            assert(tc.check(arrayfun(@(ii)var(reshape(s_p(:,:,:,ii),[1 40*40*1])) >= 0.01,1:50)));
            
            if exist('display','var') && (display == true)
                figure();
                imshow(utils.format_as_tiles(s_p,5,10));
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
            
            t = transforms.image.patch_extract(s,50,40,40,0.01,log);
            s_p = t.code(s,log);
            
            assert(tc.tensor(s_p,4));
            assert(tc.same(size(s_p),[40 40 3 50]));
            assert(tc.unitreal(s_p));
            assert(tc.check(arrayfun(@(ii)var(reshape(s_p(:,:,:,ii),[1 40*40*3])) >= 0.01,1:50)));
            
            if exist('display','var') && (display == true)
                figure();
                imshow(utils.format_as_tiles(s_p,5,10));
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
