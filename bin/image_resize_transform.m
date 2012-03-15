classdef image_resize_transform < transform
    properties (GetAccess=public,SetAccess=immutable)
        new_row_count;
        new_col_count;
    end
    
    methods (Access=public)
        function [obj] = image_resize_transform(new_row_count,new_col_count)
            assert(tc.scalar(new_row_count) && tc.natural(new_row_count) && (new_row_count > 0));
            assert(tc.scalar(new_col_count) && tc.natural(new_col_count) && (new_col_count > 0));
            
            obj.new_row_count = new_row_count;
            obj.new_col_count = new_col_count;
        end
        
        function [new_gray_images] = code(obj,gray_images)
            assert(tc.scalar(gray_images) && tc.gray_images_set(gray_images));
            
            new_gray_images_t = zeros(obj.new_row_count,obj.new_col_count,gray_images.samples_count);
            
            for i = 1:gray_images.samples_count
                new_gray_images_t(:,:,i) = imresize(gray_images.images(:,:,i),[obj.new_row_count obj.new_col_count]);
            end
            
            new_gray_images_t = utils.clamp_images_to_unit(new_gray_images_t);
            
            new_gray_images = gray_images_set(gray_images.classes,new_gray_images_t,gray_images.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "image_resize_transform".\n');
            
            fprintf('  Proper construction.\n');
            
            t = image_resize_transform(20,20);
            
            assert(t.new_row_count == 20);
            assert(t.new_col_count == 20);
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            s = gray_images_set.load_from_dir('../data/test/scenes_small');
            p_t = patch_extract_transform(80,40,40,0.0001);
            p_s = p_t.code(s);
            
            t = image_resize_transform(20,20);
            
            s_p = t.code(p_s);
            
            assert(utils.same_classes(s_p.classes,p_s.classes));
            assert(s_p.classes_count == 1);
            assert(tc.check(arrayfun(@(i)utils.approx(s_p.samples(i,:),reshape(utils.clamp_images_to_unit(imresize(p_s.images(:,:,i),[20 20])),[1 20*20])),1:80)));
            assert(tc.check(s_p.labels_idx == p_s.labels_idx));
            assert(s_p.samples_count == 80);
            assert(s_p.features_count == 20*20);
            assert(tc.check(arrayfun(@(i)utils.approx(s_p.images(:,:,i),utils.clamp_images_to_unit(imresize(p_s.images(:,:,i),[20 20]))),1:80)));
            assert(s_p.row_count == 20);
            assert(s_p.col_count == 20);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(p_s.images,8,10));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p.images,8,10));
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
