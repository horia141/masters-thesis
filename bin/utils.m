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
        
        function [tiles_image] = format_as_tiles(images,tiles_row_count,tiles_col_count)
            assert(tc.tensor(images,3) && tc.unitreal(images));
            assert(~exist('tiles_row_count','var') || ...
                   (tc.scalar(tiles_row_count) && tc.natural(tiles_row_count) && (tiles_row_count > 1)));
            assert(~exist('tiles_col_count','var') || ...
                   (tc.scalar(tiles_col_count) && tc.natural(tiles_col_count) && (tiles_col_count > 1)));
            assert(~(exist('tiles_row_count','var') && exist('tiles_col_count','var')) || ...
                     (tiles_row_count * tiles_col_count >= size(images,3)));
            assert(~(exist('tiles_row_count','var') && ~exist('tiles_col_count','var')) || ...
                     (tiles_row_count * tiles_row_count >= size(images,3)));
            
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
        function test
            fprintf('Testing "utils".\n');
            
            fprintf('  Testing function "force_row".\n');
            
            assert(all(utils.force_row([1 2 3]) == [1 2 3]));
            assert(all(utils.force_row([1;2;3]) == [1 2 3]));
            assert(all(utils.force_row(zeros(1,45)) == zeros(1,45)));
            assert(all(utils.force_row(ones(41,1)) == ones(1,41)));
            
            fprintf('  Testing function "force_col".\n');
            
            assert(all(utils.force_col([1 2 3]) == [1;2;3]));
            assert(all(utils.force_col([1;2;3]) == [1;2;3]));
            assert(all(utils.force_col(zeros(1,45)) == zeros(45,1)));
            assert(all(utils.force_col(ones(41,1)) == ones(41,1)));
            
            fprintf('  Testing function "approx".\n');
            
            t = rand(100,100);
            
            assert(utils.approx(1,1) == true);
            assert(utils.approx(1,1 + 1e-7) == true);
            assert(utils.approx(t,sqrt(t .^ 2)) == true);
            assert(utils.approx(t,(t - repmat(mean(t,1),100,1)) + repmat(mean(t,1),100,1)) == true);
            assert(utils.approx(1,1.5,0.9) == true);
            assert(utils.approx(2,2.5) == false);
            assert(utils.approx(1,1 + 1e-7,1e-9) == false);
            
            clear all;
            
            fprintf('  Testing "format_as_tiles".\n');
            
            fprintf('    Testing with unspecified "tiles_row_count" and "tiles_col_count".\n');
            
            t = rand(20,20,36);
            tt = utils.format_as_tiles(t);
            
            imshow(tt);
            
            pause(5);
            
            close(gcf());
            clear all;
            
            fprintf('    Testing with unspecified "tiles_col_count".\n');
            
            t = rand(20,20,36);
            tt = utils.format_as_tiles(t,7);
            
            imshow(tt);
            
            pause(5);
            
            close(gcf());
            clear all;
            
            fprintf('    Testing with both "tiles_row_count" and "tiles_col_count" specified.\n');
            
            t = rand(20,20,36);
            tt = utils.format_as_tiles(t,5,8);
            
            imshow(tt);

            pause(5);
            
            close(gcf());
            clear all;
            
            fprintf('  Testing "remap_images_to_unit".\n');
            
            fprintf('    Testing with mode "local" (default).\n');
            
            t = 4 * rand(20,20,36) - 2;
            t(:,:,12:18) = 4 * t(:,:,12:18);
            tp = utils.remap_images_to_unit(t);
            
            assert(tc.unitreal(tp));
            
            imshow(utils.format_as_tiles(tp));
            
            pause(5);
            
            close(gcf());
            clear all;
            
            fprintf('    Testing with mode "local".\n');
            
            t = 4 * rand(20,20,36) - 2;
            t(:,:,12:18) = 4 * t(:,:,12:18);
            tp = utils.remap_images_to_unit(t,'local');
            
            assert(tc.unitreal(tp));
            
            imshow(utils.format_as_tiles(tp));
            
            pause(5);
            
            close(gcf());
            clear all;
            
            fprintf('    Testing with mode "global".\n');
            
            t = 4 * rand(20,20,36) - 2;
            t(:,:,12:18) = 4 * t(:,:,12:18);
            tp = utils.remap_images_to_unit(t,'global');
            
            assert(tc.unitreal(tp));
            
            imshow(utils.format_as_tiles(tp));
            
            pause(5);
            
            close(gcf());
            clear all;
        end
    end
end