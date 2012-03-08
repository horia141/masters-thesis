classdef patch_extract_transform < transform
    properties (GetAccess=public,SetAccess=immutable)
        patches_count;
        patch_row_count;
        patch_col_count;
        required_variance;
    end
    
    methods (Access=public)
        function [obj] = patch_extract_transform(patches_count,patch_row_count,patch_col_count,required_variance)
            assert(tc.scalar(patches_count) && tc.natural(patches_count) && (patches_count >= 1));
            assert(tc.scalar(patch_row_count) && tc.natural(patch_row_count) && (patch_row_count >= 1));
            assert(tc.scalar(patch_col_count) && tc.natural(patch_col_count) && (patch_col_count >= 1));
            assert(tc.scalar(required_variance) && tc.number(required_variance) && (required_variance > 0));
            
            obj.patches_count = patches_count;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.required_variance = required_variance;
        end

        function [new_gray_images] = code(obj,gray_images)
            assert(tc.scalar(gray_images) && tc.gray_images_set(gray_images));
            assert(obj.patch_row_count <= gray_images.row_count);
            assert(obj.patch_col_count <= gray_images.col_count);
            
            new_gray_images_t = zeros(obj.patch_row_count,obj.patch_col_count,obj.patches_count);
            
            curr_patches_count = 1;
            
            while curr_patches_count <= obj.patches_count
                image_idx = randi(gray_images.samples_count);
                row_skip = randi(gray_images.row_count - obj.patch_row_count + 1);
                col_skip = randi(gray_images.col_count - obj.patch_col_count + 1);
                
                patch = gray_images.images(row_skip:(row_skip + obj.patch_row_count - 1),...
                                           col_skip:(col_skip + obj.patch_col_count - 1),...
                                           image_idx);
                
                if var(patch(:)) > obj.required_variance
                    new_gray_images_t(:,:,curr_patches_count) = patch;
                    curr_patches_count = curr_patches_count + 1;
                end
            end
            
            new_gray_images = gray_images_set({'none'},new_gray_images_t,ones(obj.patches_count,1));
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "patch_extract_transform".\n');
            
            fprintf('  Pproper construction.\n');
            
            t = patch_extract_transform(10,5,5,0.3);
            
            assert(t.patches_count == 10);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.required_variance == 0.3);
            
            fprintf('  Function "code".\n');
            
            s = gray_images_set.load_from_dir('../data/test');
            t = patch_extract_transform(50,40,40,0.0001);
            
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes(1),'none'));
            assert(s_p.classes_count == 1);
            assert(all(size(s_p.samples) == [50 1600]));
            assert(tc.unitreal(s_p.samples));
            assert(length(s_p.labels_idx) == 50);
            assert(all(s_p.labels_idx == ones(50,1)));
            assert(s_p.samples_count == 50);
            assert(s_p.features_count == 1600);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            assert(tc.tensor(s_p.images,3) && tc.unitreal(s_p.images));

            if exist('display','var') && (display == true)
                figure();
                imshow(utils.format_as_tiles(s_p.images,5,10));
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
