classdef remove_pinknoise_transform < transform
    methods (Access=public)
        function [obj] = remove_pinknoise_transform()
        end
        
        function [new_gray_images] = code(obj,gray_images)
            assert(tc.scalar(gray_images) && tc.gray_images_set(gray_images));
            assert(gray_images.row_count == gray_images.col_count);
            
            N = gray_images.row_count;
            
            [fx fy] = meshgrid((-N/2):(N/2-1),(-N/2):(N/2 - 1));
            rho = sqrt(fx .* fx + fy .* fy);
            f_0 = 0.4 * N;
            filter_kernel = fftshift(rho .* exp(-(rho/f_0) .^ 4));
            
            new_gray_images_t1 = zeros(size(gray_images.images));
            
            for i = 1:gray_images.samples_count
                imagew = real(ifft2(fft2(gray_images.images(:,:,i)) .* filter_kernel));
                new_gray_images_t1(:,:,i) = imagew;
            end
            
            new_gray_images_t2 = utils.remap_images_to_unit(new_gray_images_t1);
            new_gray_images = gray_images_set(gray_images.classes,new_gray_images_t2,gray_images.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "remove_pinknoise_transform".\n');
            
            fprintf('  Proper construction.\n');
            
            t = remove_pinknoise_transform();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            s = gray_images_set.load_from_dir('../data/test');
            t = remove_pinknoise_transform();
            
            p_t = patch_extract_transform(80,40,40,0.0001);
            p_s = p_t.code(s);
            
            s_p = t.code(p_s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes(1),'none'));
            assert(s_p.classes_count == 1);
            assert(all(size(s_p.samples) == [80 1600]));
            assert(tc.unitreal(s_p.samples));
            assert(length(s_p.labels_idx) == 80);
            assert(all(s_p.labels_idx == ones(80,1)));
            assert(s_p.samples_count == 80);
            assert(s_p.features_count == 1600);
            assert(s_p.row_count == 40);
            assert(s_p.col_count == 40);
            assert(tc.tensor(s_p.images,3) && tc.unitreal(s_p.images));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(2,1,1);
                imshow(utils.format_as_tiles(p_s.images,8,10));
                subplot(2,1,2);
                imshow(utils.format_as_tiles(s_p.images,8,10));
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
