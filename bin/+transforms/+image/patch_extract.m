classdef patch_extract < transform
    properties (GetAccess=public,SetAccess=immutable)
        patches_count;
        patch_row_count;
        patch_col_count;
        required_variance;
    end
    
    methods (Access=public)
        function [obj] = patch_extract(train_sample_plain,patches_count,patch_row_count,patch_col_count,required_variance)
            assert(check.dataset_image(train_sample_plain));
            assert(check.scalar(patches_count));
            assert(check.natural(patches_count));
            assert(patches_count >= 1);
            assert(check.scalar(patch_row_count));
            assert(check.natural(patch_row_count));
            assert(patch_row_count >= 1);
            assert(check.scalar(patch_col_count));
            assert(check.natural(patch_col_count));
            assert(patch_col_count >= 1);
            assert(check.scalar(required_variance));
            assert(check.number(required_variance));
            assert(required_variance >= 0);
            
            [d dr dc dl] = dataset.geometry(train_sample_plain);
            
            input_geometry = [d dr dc dl];
            output_geometry = [patch_row_count*patch_col_count*dl patch_row_count patch_col_count dl];
            
            obj = obj@transform(input_geometry,output_geometry);
            obj.patches_count = patches_count;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.required_variance = required_variance;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain)
            N = dataset.count(sample_plain);
            [~,dr,dc,dl] = dataset.geometry(sample_plain);
            
            log_batch_size = ceil(obj.patches_count / 10);
            sample_coded = zeros(obj.patch_row_count,obj.patch_col_count,dl,obj.patches_count);
            
            curr_patches_count = 1;
            
            while curr_patches_count <= obj.patches_count
                image_idx = randi(N);
                row_skip = randi(dr - obj.patch_row_count + 1);
                col_skip = randi(dc - obj.patch_col_count + 1);
                
                patch = sample_plain(row_skip:(row_skip + obj.patch_row_count - 1),...
                                     col_skip:(col_skip + obj.patch_col_count - 1),...
                                     :,image_idx);
                
                if var(patch(:)) >= obj.required_variance
                    sample_coded(:,:,:,curr_patches_count) = patch;
                    curr_patches_count = curr_patches_count + 1;
                end
            end
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.image.patch_extract".\n');
            
            fprintf('  Proper construction.\n');
            
            s = dataset.load('../test/scenes_small.mat');
            
            t = transforms.image.patch_extract(s,10,5,5,0.01);
            
            assert(t.patches_count == 10);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.required_variance == 0.01);
            assert(check.same(t.input_geometry,[192*256*3 192 256 3]));
            assert(check.same(t.output_geometry,[5*5*3 5 5 3]));
            
            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            s = dataset.load('../test/scenes_small.mat');
            
            t = transforms.image.patch_extract(s,50,40,40,0.01);
            s_p = t.code(s);
            
            assert(check.tensor(s_p,4));
            assert(check.same(size(s_p),[40 40 3 50]));
            assert(check.unitreal(s_p));
            assert(check.checkf(@(ii)var(reshape(s_p(:,:,:,ii),[1 40*40*3])) >= 0.01,1:50));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,1,1);
                utils.display.as_tiles(s_p,[5 10]);
                pause(5);
            end
            
            clearvars -except test_figure;
        end
    end
end
