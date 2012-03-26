classdef window < transform
    properties (GetAccess=public,SetAccess=immutable)
        filter;
        sigma;
    end
    
    methods (Access=public)
        function [obj] = window(train_image_plain,sigma)
            assert(tc.scalar(train_image_plain) && tc.datasets_image(train_image_plain));
            assert(train_image_plain.samples_count >= 1);
            assert(train_image_plain.row_count == train_image_plain.col_count);
            assert(tc.scalar(sigma) && tc.number(sigma) && (sigma > 0));
            
            filter_t = fspecial('gaussian',train_image_plain.row_count,sigma);
            filter_t = filter_t ./ max(max(filter_t));
            
            obj = obj@transform(train_image_plain.subsamples(1));
            obj.filter = filter_t;
            obj.sigma = sigma;
        end
    end
    
    methods (Access=protected)
        function [image_coded] = do_code(obj,image_plain)
            images_coded = zeros(image_plain.row_count,image_plain.col_count,image_plain.layers_count,image_plain.samples_count);
            
            for layer = 1:image_plain.layers_count
                for i = 1:image_plain.samples_count
                    images_coded(:,:,layer,i) = image_plain.images(:,:,layer,i) .* obj.filter;
                end
            end
            
            image_coded = datasets.image(image_plain.classes,images_coded,image_plain.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.window".\n');
            
            fprintf('  Proper construction.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','gray',[192 192]);
            
            t = transforms.image.window(s,10);
            
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
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','gray',[192 192]);
            
            t = transforms.image.window(s,60);
            s_p = t.code(s);
            
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
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s.images,4,2));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p.images,4,2));
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','color',[192 192]);
            
            t = transforms.image.window(s,60);
            s_p = t.code(s);
            
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
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s.images,4,2));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p.images,4,2));
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end