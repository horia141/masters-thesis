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
        function [obj] = patch_extract(train_image_plain,patches_count,patch_row_count,patch_col_count,required_variance)
            assert(tc.scalar(train_image_plain) && tc.datasets_image(train_image_plain));
            assert(train_image_plain.samples_count >= 1);
            assert(tc.scalar(patches_count) && tc.natural(patches_count) && (patches_count >= 1));
            assert(tc.scalar(patch_row_count) && tc.natural(patch_row_count) && (patch_row_count >= 1));
            assert(tc.scalar(patch_col_count) && tc.natural(patch_col_count) && (patch_col_count >= 1));
            assert(tc.scalar(required_variance) && tc.number(required_variance) && (required_variance >= 0));
            
            obj = obj@transform();
            obj.patches_count = 1;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.required_variance = required_variance;
            obj.one_sample_plain = train_image_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain);
            obj.patches_count = patches_count; % HACK DI HACK HACK %
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
            
            fprintf('  Proper construction.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.patch_extract(s,10,5,5,0.01);
            
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
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            t = transforms.image.patch_extract(s,50,40,40,0.01);
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
            assert(tc.check(arrayfun(@(i)var(reshape(s_p.images(:,:,:,i),[1 40*40])) >= 0.01,1:50)));
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
            
            t = transforms.image.patch_extract(s,50,40,40,0.01);
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
            assert(tc.check(arrayfun(@(i)var(reshape(s_p.images(:,:,:,i),[1 3*40*40])) >= 0.01,1:50)));
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
