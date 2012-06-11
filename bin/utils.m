classdef utils
    methods (Static,Access=public)
        function [o] = rand_range(a,b,varargin)
            assert(tc.scalar(a));
            assert(tc.number(a));
            assert(tc.scalar(b));
            assert(tc.scalar(b));
            assert(tc.empty(varargin) || tc.vector(varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.scalar,varargin));
            assert(tc.empty(varargin) || tc.checkf(@tc.number,varargin));
            assert(a < b);
            
            o = (b - a) * rand(varargin{:}) + a;
        end
        
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
        
        function [o_v] = force_same(v,other_v)
            assert(tc.vector(v));
            assert(tc.vector(other_v));
            
            if size(other_v,2) == length(other_v)
                o_v = utils.force_row(v);
            else
                o_v = utils.force_col(v);
            end
        end
        
        function [tiles_image] = format_as_tiles(images,tiles_row_count,tiles_col_count)
            assert(tc.dataset_image(images));
            assert(~exist('tiles_row_count','var') || tc.scalar(tiles_row_count));
            assert(~exist('tiles_row_count','var') || tc.natural(tiles_row_count));
            assert(~exist('tiles_row_count','var') || (tiles_row_count >= 1));
            assert(~exist('tiles_col_count','var') || tc.scalar(tiles_col_count));
            assert(~exist('tiles_col_count','var') || tc.natural(tiles_col_count));
            assert(~exist('tiles_col_count','var') || (tiles_col_count >= 1));
            assert(~(exist('tiles_row_count','var') && exist('tiles_col_count','var')) || ...
                     (tiles_row_count * tiles_col_count >= size(images,4)));
            assert(~(exist('tiles_row_count','var') && ~exist('tiles_col_count','var')) || ...
                     (tiles_row_count * tiles_row_count >= size(images,4)));
            
            images_count = size(images,4);
            layers_count = size(images,3);
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
                               (col_count + 1) * tiles_col_count_t + 1,...
                               layers_count);
    
            for layer = 1:layers_count
                for ii = 1:tiles_row_count_t
                    for jj = 1:tiles_col_count_t
                        if (ii - 1) * tiles_col_count_t + jj <= images_count
                            image = images(:,:,layer,(ii - 1) * tiles_col_count_t + jj);
                            tiles_image(((ii - 1)*(row_count + 1) + 2):(ii*(row_count + 1)),...
                                        ((jj - 1)*(col_count + 1) + 2):(jj*(col_count + 1)),layer) = image;
                        else
                            tiles_image(((ii - 1)*(row_count + 1) + 2):(ii*(row_count + 1)),...
                                        ((jj - 1)*(col_count + 1) + 2):(jj*(col_count + 1)),layer) = 0.25 * ones(row_count,col_count);
                        end
                    end
                end
            end
        end
        
        function [new_images] = clamp_images_to_unit(images)
            assert(tc.dataset_image(images));
            
            new_images = max(min(images,1),0);
        end
        
        function [new_images] = remap_images_to_unit(images,mode)
            assert(tc.dataset_image(images));
            assert(~exist('mode','var') || tc.scalar(mode));
            assert(~exist('mode','var') || tc.string(mode));
            assert(~exist('mode','var') || tc.one_of(mode,'local','global'));

            if exist('mode','var')
                mode_t = mode;
            else
                mode_t = 'local';
            end
            
            images_count = size(images,4);
            layers_count = size(images,3);
            row_count = size(images,1);
            col_count = size(images,2);
            
            new_images = zeros(row_count,col_count,layers_count,images_count);
            
            if strcmp(mode_t,'local')
                for layer = 1:layers_count
                    for ii = 1:images_count
                        im_min = min(min(images(:,:,layer,ii)));
                        im_max = max(max(images(:,:,layer,ii)));
                    
                        new_images(:,:,layer,ii) = (images(:,:,layer,ii) - im_min) / (im_max - im_min);
                    end
                end
            else
                im_min = min(images(:));
                im_max = max(images(:));
                
                new_images = (images - im_min) / (im_max - im_min);
            end
        end
        
        function [s] = value_to_string(i)
            assert(tc.scalar(i));
            assert(tc.value(i));
            
            if tc.logical(i)
                if i
                    s = 'true';
                else
                    s = 'false';
                end
            elseif tc.number(i)
                if tc.integer(i)
                    s = sprintf('%d',i);
                else
                    s = sprintf('%f',i);
                end
            elseif tc.string(i)
                s = i;
            elseif tc.function_h(i)
                s = func2str(i);
            else
                assert(false);
            end
        end
        
        function [s] = matrix_to_string(mat,format)
            assert(tc.matrix(mat));
            assert(tc.number(mat));
            assert(~exist('format','var') || tc.scalar(format));
            assert(~exist('format','var') || tc.string(format));

            if exist('format','var')
                format_t = format;
            else
                format_t = '%f ';
            end
            
            contents = cell(size(mat));
            
            for ii = 1:size(mat,1)
                for jj = 1:size(mat,2)
                    contents{ii,jj} = sprintf(format_t,mat(ii,jj));
                end
            end
            
            max_size = max(cellfun(@length,contents(:)));
            size_format = sprintf('%%%ds',max_size);
                
            s = '';
            
            for ii = 1:size(mat,1)
                for jj = 1:size(mat,2)
                    s = sprintf('%s%s',s,sprintf(size_format,contents{ii,jj}));
                end
                
                s = sprintf('%s\n',s);
            end
        end
        
        function [o] = cell_cull(cells)
            assert(tc.vector(cells));
            assert(tc.cell(cells));
            
            o = {};
            
            for ii = 1:length(cells)
                if ~tc.empty(cells{ii})
                    o = [o; cells(ii)];
                end
            end
            
            if ~tc.empty(o)
                o = utils.force_same(o,cells);
            end
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "utils".\n');
            
            fprintf('  Function "rand_range".\n');
            
            r1 = utils.rand_range(0,1);
            r2 = utils.rand_range(-1,1);
            r3 = utils.rand_range(0,10,1,10);
            r4 = utils.rand_range(-3,3,4,4,5);
            
            assert(tc.scalar(r1));
            assert(tc.unitreal(r1));
            assert(tc.scalar(r2));
            assert(tc.unitreal(abs(r2)));
            assert(tc.vector(r3));
            assert(length(r3) == 10);
            assert(tc.check(r3 >= 0));
            assert(tc.check(r3 <= 10));
            assert(tc.tensor(r4,3));
            assert(size(r4,1) == 4);
            assert(size(r4,2) == 4);
            assert(size(r4,3) == 5);
            assert(tc.check(r4 >= -3));
            assert(tc.check(r4 <= 3));
            
            fprintf('  Function "force_row".\n');
            
            assert(tc.same(utils.force_row(1),1));
            assert(tc.same(utils.force_row([1 2 3]),[1 2 3]));
            assert(tc.same(utils.force_row([1;2;3]),[1 2 3]));
            assert(tc.same(utils.force_row(zeros(1,45)),zeros(1,45)));
            assert(tc.same(utils.force_row(ones(41,1)),ones(1,41)));
            assert(tc.same(utils.force_row({1 2 3}),{1 2 3}));
            assert(tc.same(utils.force_row({1;2;3}),{1 2 3}));
            assert(tc.same(utils.force_row({'hello';'world'}),{'hello' 'world'}));
            
            clearvars -except display;
            
            fprintf('  Function "force_col".\n');

            assert(tc.same(utils.force_col(1),1));
            assert(tc.same(utils.force_col([1 2 3]),[1;2;3]));
            assert(tc.same(utils.force_col([1;2;3]),[1;2;3]));
            assert(tc.same(utils.force_col(zeros(1,45)),zeros(45,1)));
            assert(tc.same(utils.force_col(ones(41,1)),ones(41,1)));
            assert(tc.same(utils.force_col({1 2 3}),{1;2;3}));
            assert(tc.same(utils.force_col({1;2;3}),{1;2;3}));
            assert(tc.same(utils.force_col({'hello' 'world'}),{'hello';'world'}));
            
            fprintf('  Function "force_same".\n');
            
            assert(tc.same(utils.force_same(1,true),1));
            assert(tc.same(utils.force_same([1 2 3],[true true false]),[1 2 3]));
            assert(tc.same(utils.force_same([1 2 3],[true;true;false]),[1;2;3]));
            assert(tc.same(utils.force_same([1;2],[true true false]),[1 2]));
            assert(tc.same(utils.force_same([1;2],[true;true;false]),[1;2]));
            
            clearvars -except display;
            
            fprintf('  Function "format_as_tiles".\n');
            
            fprintf('    With unspecified "tiles_row_count" and "tiles_col_count".\n');
            
            t = rand(20,20,1,36);
            tt = utils.format_as_tiles(t);
            
            if exist('display','var') && (display == true)
                figure();
                imshow(tt);
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With unspecified "tiles_col_count".\n');
            
            t = rand(20,20,1,36);
            tt = utils.format_as_tiles(t,7);
            
            if exist('display','var') && (display == true)
                figure();
                imshow(tt);
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With both "tiles_row_count" and "tiles_col_count" specified.\n');
            
            t = rand(20,20,1,36);
            tt = utils.format_as_tiles(t,5,8);
            
            if exist('display','var') && (display == true)
                figure();
                imshow(tt);
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With relaxed input "images" not "unitreal".\n');
            
            t = 3*rand(20,20,1,36) - 1.5;
            tt = utils.format_as_tiles(t,6,6);
            
            if exist('display','var') && (display == true)
                figure();
                imshow(tt);
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            t = rand(20,20,3,36);
            tt = utils.format_as_tiles(t);
            
            if exist('display','var') && (display == true)
                figure();
                imshow(tt);
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "clamp_images_to_unit".\n');
            
            fprintf('    With small excursions below 0 and above 1.\n');
            
            t = rand(20,20,1,36) + 0.2;
            tp = utils.clamp_images_to_unit(t);
            
            assert(min(tp(:)) >= 0);
            assert(max(tp(:)) <= 1);
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Clamped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With large excursions below 0 and above 1.\n');
            
            t = 4 * rand(20,20,1,36) - 2;
            tp = utils.clamp_images_to_unit(t);
            
            assert(min(tp(:)) >= 0);
            assert(max(tp(:)) <= 1);
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Clamped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            t = rand(20,20,3,36) + 0.2;
            tp = utils.clamp_images_to_unit(t);
            
            assert(min(tp(:)) >= 0);
            assert(max(tp(:)) <= 1);
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6));
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
            
            t = 4 * rand(20,20,1,36) - 2;
            t(:,:,12:18) = 4 * t(:,:,12:18);
            tp = utils.remap_images_to_unit(t);
            
            assert(tc.same(min(tp(:)),0));
            assert(tc.same(max(tp(:)),1));
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Remaped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With mode "local".\n');
            
            t = 4 * rand(20,20,1,36) - 2;
            t(:,:,12:18) = 4 * t(:,:,12:18);
            tp = utils.remap_images_to_unit(t,'local');
            
            assert(tc.same(min(tp(:)),0));
            assert(tc.same(max(tp(:)),1));
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Remaped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With mode "global".\n');
            
            t = 4 * rand(20,20,1,36) - 2;
            t(:,:,12:18) = 4 * t(:,:,12:18);
            tp = utils.remap_images_to_unit(t,'global');
            
            assert(tc.same(min(tp(:)),0));
            assert(tc.same(max(tp(:)),1));
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Remaped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With color images.\n');
            
            t = 4 * rand(20,20,3,36) - 2;
            t(:,:,12:18) = 4 * t(:,:,12:18);
            tp = utils.remap_images_to_unit(t);
            
            assert(tc.same(min(tp(:)),0));
            assert(tc.same(max(tp(:)),1));
            assert(tc.unitreal(tp));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(t,6,6));
                title('Problem images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(tp));
                title('Remaped images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "value_to_string".\n');
            
            assert(tc.same(utils.value_to_string(true),'true'));
            assert(tc.same(utils.value_to_string(false),'false'));
            assert(tc.same(utils.value_to_string(7.3),'7.300000'));
            assert(tc.same(utils.value_to_string(-5),'-5'));
            assert(tc.same(utils.value_to_string(7),'7'));
            assert(tc.same(utils.value_to_string('hello'),'hello'));
            assert(tc.same(utils.value_to_string(@(c)c),'@(c)c'));
            assert(tc.same(utils.value_to_string(@transforms.image.random_corr.max),'transforms.image.random_corr.max'));
            
            clearvars -except display;
            
            fprintf('  Function "matrix_to_string".\n');
            
            assert(tc.same(utils.matrix_to_string([1 0; 0 1]),sprintf('1.000000 0.000000 \n0.000000 1.000000 \n')));
            assert(tc.same(utils.matrix_to_string([1 2 3; 4 5 6; 7 8 9]),sprintf('1.000000 2.000000 3.000000 \n4.000000 5.000000 6.000000 \n7.000000 8.000000 9.000000 \n')));
            assert(tc.same(utils.matrix_to_string([1 0; 0 1],'%d '),sprintf('1 0 \n0 1 \n')));
            assert(tc.same(utils.matrix_to_string([1.2 3.2; 4.4 1.3],' %.2f '),sprintf(' 1.20  3.20 \n 4.40  1.30 \n')));
            
            clearvars -except display;
            
            fprintf('  Function "cell_cull".\n');
            
            assert(tc.same(utils.cell_cull({1 2 3}),{1 2 3}));
            assert(tc.same(utils.cell_cull({1 {} 3}),{1 3}));
            assert(tc.same(utils.cell_cull({{} {}}),{}));
            assert(tc.same(utils.cell_cull({1;2;3}),{1;2;3}));
            assert(tc.same(utils.cell_cull({1;{};3}),{1;3}));
            assert(tc.same(utils.cell_cull({{};{}}),{}));
        end
    end
end
