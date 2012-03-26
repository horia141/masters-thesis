classdef patch_extract < transform
    properties (GetAccess=public,SetAccess=immutable)
        patches_count;
        patch_row_count;
        patch_col_count;
        required_variance;
    end
    
    methods (Access=public)
        function [obj] = patch_extract(train_image_plain,patches_count,patch_row_count,patch_col_count,required_variance)
            assert(tc.scalar(train_image_plain) && tc.datasets_image(train_image_plain));
            assert(train_image_plain.samples_count >= 1);
            assert(tc.scalar(patches_count) && tc.natural(patches_count) && (patches_count >= 1));
            assert(tc.scalar(patch_row_count) && tc.natural(patch_row_count) && (patch_row_count >= 1));
            assert(tc.scalar(patch_col_count) && tc.natural(patch_col_count) && (patch_col_count >= 1));
            assert(tc.scalar(required_variance) && tc.number(required_variance) && (required_variance >= 0));
            
            obj = obj@transform(train_image_plain.subsamples(1));
            obj.patches_count = patches_count;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.required_variance = required_variance;
        end
    end
    
    methods (Access=protected)
        function [image_coded] = do_code(obj,image_plain)
            images_coded = zeros(obj.patch_row_count,obj.patch_col_count,image_plain.layers_count,obj.patches_count);
            
            curr_patches_count = 1;
            
            while curr_patches_count <= obj.patches_count
                image_idx = randi(image_plain.samples_count);
                row_skip = randi(image_plain.row_count - obj.patch_row_count + 1);
                col_skip = randi(image_plain.col_count - obj.patch_col_count + 1);
                
                patch = image_plain.images(row_skip:(row_skip + obj.patch_row_count - 1),...
                                           col_skip:(col_skip + obj.patch_col_count - 1),...
                                           :,image_idx);
                
                if var(patch(:)) > obj.required_variance
                    images_coded(:,:,:,curr_patches_count) = patch;
                    curr_patches_count = curr_patches_count + 1;
                end
            end
            
            image_coded = datasets.image({'none'},images_coded,ones(obj.patches_count,1));
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.patch_extract".\n');
            
            fprintf('  Pproper construction.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.patch_extract(s,10,5,5,0.3);
            
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
            assert(t.patches_count == 10);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.required_variance == 0.3);
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.patch_extract(s,50,40,40,0.0001);
            s_p = t.code(s);
            
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
            assert(s_p.layers_count == 1);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);            

            if exist('display','var') && (display == true)
                figure();
                imshow(utils.format_as_tiles(s_p.images,5,10));
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','color');
            
            t = transforms.image.patch_extract(s,50,40,40,0.0001);
            s_p = t.code(s);
            
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
            assert(s_p.layers_count == 3);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);            

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
