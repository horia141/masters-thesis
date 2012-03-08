classdef utils
    methods (Static,Access=public)
        function [o_v] = force_row(v)
            assert(tc.vector(v));
            
            if size(v,2) == length(v)
                o_v = v;
            else
                o_v = v';
            end
        end
        
        function [o_v] = force_col(v)
            assert(tc.vector(v));
            
            if size(v,1) == length(v)
                o_v = v;
            else
                o_v = v';
            end
        end
        
        function [o] = approx(v1,v2,epsilon)
            assert(tc.number(v1));
            assert(tc.number(v2));
            assert(~exist('epsilon','var') || tc.unitreal(epsilon));
            
            if exist('epsilon','var')
                epsilon_t = epsilon;
            else
                epsilon_t = 1e-6;
            end
            
            r = abs(v1 - v2) < epsilon_t;
            o = all(r(:));
        end
        
        function [tiles_image] = format_as_tiles(images,tiles_row_count,tiles_col_count,bypass_unitreal)
            assert(tc.tensor(images,3) && ...
                   (tc.unitreal(images) || (exist('bypass_unitreal','var') && tc.number(images))));
            assert(~exist('tiles_row_count','var') || ...
                   (tc.scalar(tiles_row_count) && tc.natural(tiles_row_count) && (tiles_row_count > 1)));
            assert(~exist('tiles_col_count','var') || ...
                   (tc.scalar(tiles_col_count) && tc.natural(tiles_col_count) && (tiles_col_count > 1)));
            assert(~(exist('tiles_row_count','var') && exist('tiles_col_count','var')) || ...
                     (tiles_row_count * tiles_col_count >= size(images,3)));
            assert(~(exist('tiles_row_count','var') && ~exist('tiles_col_count','var')) || ...
                     (tiles_row_count * tiles_row_count >= size(images,3)));
            assert(~exist('bypas_unitreal','var') || (tc.scalar(bypass_unitreal) && bypass_unitreal));
            
            images_count = size(images,3);
            row_count = size(images,1);
            col_count = size(images,2);
            
            if exist('tiles_row_count','var') && exist('tiles_col_count','var')
                tiles_row_count_t = tiles_row_count;
                tiles_col_count_t = tiles_col_count;
            elseif exist('tiles_row_count','var') && ~exist('tiles_col_count','var')
                tiles_row_count_t = tiles_row_count;
                tiles_col_count_t = tiles_row_count;
            else
                tiles_row_count_t = ceil(sqrt(images_count));
                tiles_col_count_t = ceil(sqrt(images_count));
            end
    
            tiles_image = ones((row_count + 1) * tiles_row_count_t + 1,...
                               (col_count + 1) * tiles_col_count_t + 1);
    
            for i = 1:tiles_row_count_t
                for j = 1:tiles_col_count_t
                    if (i - 1) * tiles_col_count_t + j <= images_count
                        image = images(:,:,(i - 1) * tiles_col_count_t + j);
                        tiles_image(((i - 1)*(row_count + 1) + 2):(i*(row_count + 1)),...
                                    ((j - 1)*(col_count + 1) + 2):(j*(col_count + 1))) = image;
                    else
                        tiles_image(((i - 1)*(row_count + 1) + 2):(i*(row_count + 1)),...
                                    ((j - 1)*(col_count + 1) + 2):(j*(col_count + 1))) = 0.25 * ones(row_count,col_count);
                    end
                end
            end
        end
        
        function [new_images] = clamp_images_to_unit(images)
            assert(tc.tensor(images,3) && tc.number(images));
            
            new_images = max(min(images,1),0);
        end
        
        function [new_images] = remap_images_to_unit(images,mode)
            assert(tc.tensor(images,3) && tc.number(images));
            assert(~exist('mode','var') || (tc.string(mode) && ...
                     (strcmp(mode,'local') || strcmp(mode,'global'))));
                 
            if exist('mode','var')
                mode_t = mode;
            else
                mode_t = 'local';
            end
            
            images_count = size(images,3);
            row_count = size(images,1);
            col_count = size(images,2);
            
            new_images = zeros(row_count,col_count,images_count);
            
            if strcmp(mode_t,'local')
                for i = 1:images_count
                    im_min = min(min(images(:,:,i)));
                    im_max = max(max(images(:,:,i)));
                    
                    new_images(:,:,i) = (images(:,:,i) - im_min) / (im_max - im_min);
                end
            else
                im_min = min(images(:));
                im_max = max(images(:));
                
                new_images = (images - im_min) / (im_max - im_min);
            end
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "utils".\n');
            
            fprintf('  Function "force_row".\n');
            
            assert(all(utils.force_row([1 2 3]) == [1 2 3]));
            assert(all(utils.force_row([1;2;3]) == [1 2 3]));
            assert(all(utils.force_row(zeros(1,45)) == zeros(1,45)));
            assert(all(utils.force_row(ones(41,1)) == ones(1,41)));
            
            fprintf('  Function "force_col".\n');
            
            assert(all(utils.force_col([1 2 3]) == [1;2;3]));
            assert(all(utils.force_col([1;2;3]) == [1;2;3]));
            assert(all(utils.force_col(zeros(1,45)) == zeros(45,1)));
            assert(all(utils.force_col(ones(41,1)) == ones(41,1)));
            
            fprintf('  Function "approx".\n');
            
            t = rand(100,100);
            
            assert(utils.approx(1,1) == true);
            assert(utils.approx(1,1 + 1e-7) == true);
            assert(utils.approx(t,sqrt(t .^ 2)) == true);
            assert(utils.approx(t,(t - repmat(mean(t,1),100,1)) + repmat(mean(t,1),100,1)) == true);
            assert(utils.approx(1,1.5,0.9) == true);
            assert(utils.approx(2,2.5) == false);
            assert(utils.approx(1,1 + 1e-7,1e-9) == false);
            
            clearvars -except display;
            
            fprintf('  Function "format_as_tiles".\n');
            
            fprintf('    With unspecified "tiles_row_count" and "tiles_col_count".\n');
            
            t = rand(20,20,36);
            tt = utils.format_as_tiles(t);
            
            if exist('display','var') && (display == true)
                figure();
                imshow(tt);
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With unspecified "tiles_col_count".\n');
            
            t = rand(20,20,36);
            tt = utils.format_as_tiles(t,7);
            
            if exist('display','var') && (display == true)
                figure();
                imshow(tt);
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With both "tiles_row_count" and "tiles_col_count" specified.\n');
            
            t = rand(20,20,36);
            tt = utils.format_as_tiles(t,5,8);
            
            if exist('display','var') && (display == true)
                figure();
                imshow(tt);
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With relaxed input ("images" does not have to be "unitreal").\n');
            
            t = 3*rand(20,20,36) - 1.5;
            tt = utils.format_as_tiles(t,6,6,true);
            
            if exist('display','var') && (display == true)
                figure();
                imshow(tt);
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "clamp_images_to_unit".\n');
            
            fprintf('    With small excursions below 0 and above 1.\n');
            
            t = rand(20,20,36) + 0.2;
            tp = utils.clamp_images_to_unit(t);
            
            assert(min(tp(:)) >= 0);
            assert(max(tp(:)) <= 1);
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6,true));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Clamped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With large excursions below 0 and above 1.\n');
            
            t = 4 * rand(20,20,36) - 2;
            tp = utils.clamp_images_to_unit(t);
            
            assert(min(tp(:)) >= 0);
            assert(max(tp(:)) <= 1);
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6,true));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Clamped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;

            fprintf('  Function "remap_images_to_unit".\n');
            
            fprintf('    With mode "local" (default).\n');
            
            t = 4 * rand(20,20,36) - 2;
            t(:,:,12:18) = 4 * t(:,:,12:18);
            tp = utils.remap_images_to_unit(t);
            
            assert(min(tp(:)) == 0);
            assert(max(tp(:)) == 1);
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6,true));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Remaped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With mode "local".\n');
            
            t = 4 * rand(20,20,36) - 2;
            t(:,:,12:18) = 4 * t(:,:,12:18);
            tp = utils.remap_images_to_unit(t,'local');
            
            assert(min(tp(:)) == 0);
            assert(max(tp(:)) == 1);
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6,true));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Remaped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With mode "global".\n');
            
            t = 4 * rand(20,20,36) - 2;
            t(:,:,12:18) = 4 * t(:,:,12:18);
            tp = utils.remap_images_to_unit(t,'global');
            
            assert(min(tp(:)) == 0);
            assert(max(tp(:)) == 1);
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6,true));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Remaped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
