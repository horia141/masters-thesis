classdef remove_pinknoise < transform        
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = remove_pinknoise(train_image_plain)
            assert(tc.scalar(train_image_plain) && tc.datasets_image(train_image_plain));
            assert(train_image_plain.samples_count >= 1);
            assert(train_image_plain.row_count == train_image_plain.col_count);
            
            obj = obj@transform();
            obj.one_sample_plain = train_image_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain);
        end
    end
    
    methods (Access=protected)
        function [image_coded] = do_code(~,image_plain)
            N = image_plain.row_count;
            
            [fx fy] = meshgrid((-N/2):(N/2-1),(-N/2):(N/2 - 1));
            rho = sqrt(fx .* fx + fy .* fy);
            f_0 = 0.4 * N;
            filter_kernel = fftshift(rho .* exp(-(rho/f_0) .^ 4));
            
            images_coded = zeros(image_plain.row_count,image_plain.col_count,image_plain.layers_count,image_plain.samples_count);
            
            for layer = 1:image_plain.layers_count
                for i = 1:image_plain.samples_count
                    imagew = real(ifft2(fft2(image_plain.images(:,:,layer,i)) .* filter_kernel));
                    images_coded(:,:,layer,i) = imagew;
                end
            end
            
            images_coded = utils.remap_images_to_unit(images_coded,'global');
            image_coded = datasets.image(image_plain.classes,images_coded,image_plain.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.remove_pinknoise".\n');
            
            fprintf('  Proper construction.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','gray',[192 192]);
            
            t = transforms.image.remove_pinknoise(s);
            
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
            assert(length(t.one_sample_coded.classes) == 1);
            assert(strcmp(t.one_sample_coded.classes{1},'none'));
            assert(t.one_sample_coded.classes_count == 1);
            assert(tc.check(size(t.one_sample_coded.samples) == [1 192*192]));
            assert(tc.matrix(t.one_sample_coded.samples) && tc.unitreal(t.one_sample_coded.samples));
            assert(tc.check(t.one_sample_coded.labels_idx == s.labels_idx(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 192*192);
            assert(tc.check(size(t.one_sample_coded.images) == [192 192]));
            assert(tc.tensor(t.one_sample_coded.images,4) && tc.unitreal(t.one_sample_coded.images));
            assert(t.one_sample_coded.layers_count == 1);
            assert(t.one_sample_coded.row_count == 192);
            assert(t.one_sample_coded.col_count == 192);
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With grayscale images.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            p_t = transforms.image.patch_extract(s,80,40,40,0.0001);
            p_s = p_t.code(s);
            
            t = transforms.image.remove_pinknoise(p_s);
            s_p = t.code(p_s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(tc.check(size(s_p.samples) == [80 1600]));
            assert(tc.matrix(s_p.samples) && tc.unitreal(s_p.samples));
            assert(tc.check(s_p.labels_idx == ones(80,1)));
            assert(s_p.samples_count == 80);
            assert(s_p.features_count == 1600);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            assert(tc.check(size(s_p.images) == [40 40 1 80]));
            assert(tc.tensor(s_p.images,4) && tc.unitreal(s_p.images));
            assert(s_p.layers_count == 1);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            
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
            
            fprintf('    With color images.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','color');
            p_t = transforms.image.patch_extract(s,80,40,40,0.0001);
            p_s = p_t.code(s);
            
            t = transforms.image.remove_pinknoise(p_s);
            s_p = t.code(p_s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(tc.check(size(s_p.samples) == [80 3*40*40]));
            assert(tc.matrix(s_p.samples) && tc.unitreal(s_p.samples));
            assert(tc.check(s_p.labels_idx == ones(80,1)));
            assert(s_p.samples_count == 80);
            assert(s_p.features_count == 3*40*40);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            assert(tc.check(size(s_p.images) == [40 40 3 80]));
            assert(tc.tensor(s_p.images,4) && tc.unitreal(s_p.images));
            assert(s_p.layers_count == 3);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            
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
