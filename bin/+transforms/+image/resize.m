classdef resize < transform
    properties (GetAccess=public,SetAccess=immutable)
        new_row_count;
        new_col_count;
    end
    
    methods (Access=public)
        function [obj] = resize(train_image_plain,new_row_count,new_col_count)
            assert(tc.scalar(train_image_plain) && tc.datasets_image(train_image_plain));
            assert(train_image_plain.samples_count >= 1);
            assert(tc.scalar(new_row_count) && tc.natural(new_row_count) && (new_row_count > 0));
            assert(tc.scalar(new_col_count) && tc.natural(new_col_count) && (new_col_count > 0));
            
            obj = obj@transform(train_image_plain.subsamples(1));
            obj.new_row_count = new_row_count;
            obj.new_col_count = new_col_count;
        end
    end
    
    methods (Access=protected)
        function [image_coded] = do_code(obj,image_plain)            
            images_coded = zeros(obj.new_row_count,obj.new_col_count,image_plain.layers_count,image_plain.samples_count);
            
            for i = 1:image_plain.samples_count
                images_coded(:,:,:,i) = imresize(image_plain.images(:,:,:,i),[obj.new_row_count obj.new_col_count]);
            end
            
            images_coded = utils.clamp_images_to_unit(images_coded);
            image_coded = datasets.image(image_plain.classes,images_coded,image_plain.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.resize".\n');
            
            fprintf('  Proper construction.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.resize(s,20,20);
            
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
            assert(t.new_row_count == 20);
            assert(t.new_col_count == 20);
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.resize(s,100,100);
            s_p = t.code(s);
            
            assert(utils.same_classes(s_p.classes,s.classes));
            assert(s_p.classes_count == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(tc.check(arrayfun(@(i)utils.approx(s_p.samples(i,:),reshape(utils.clamp_images_to_unit(imresize(s.images(:,:,:,i),[100 100])),[1 100*100])),1:7)));
            assert(tc.check(s_p.labels_idx == s.labels_idx));
            assert(s_p.samples_count == 7);
            assert(s_p.features_count == 100*100);
            assert(tc.check(arrayfun(@(i)utils.approx(s_p.images(:,:,:,i),utils.clamp_images_to_unit(imresize(s.images(:,:,:,i),[100 100]))),1:7)));
            assert(s_p.layers_count == 1);
            assert(s_p.row_count == 100);
            assert(s_p.col_count == 100);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s.images,3,3));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p.images,3,3));
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','color');
            
            t = transforms.image.resize(s,100,100);
            s_p = t.code(s);
            
            assert(utils.same_classes(s_p.classes,s.classes));
            assert(s_p.classes_count == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(tc.check(arrayfun(@(i)utils.approx(s_p.samples(i,:),reshape(utils.clamp_images_to_unit(imresize(s.images(:,:,:,i),[100 100])),[1 3*100*100])),1:7)));
            assert(tc.check(s_p.labels_idx == s.labels_idx));
            assert(s_p.samples_count == 7);
            assert(s_p.features_count == 3*100*100);
            assert(tc.check(arrayfun(@(i)utils.approx(s_p.images(:,:,:,i),utils.clamp_images_to_unit(imresize(s.images(:,:,:,i),[100 100]))),1:7)));
            assert(s_p.layers_count == 3);
            assert(s_p.row_count == 100);
            assert(s_p.col_count == 100);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s.images,3,3));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p.images,3,3));
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
