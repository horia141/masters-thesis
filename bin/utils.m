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
            assert(~exist('epsilon','var') || (tc.scalar(epsilon) && tc.unitreal(epsilon)));
            
            if exist('epsilon','var')
                epsilon_t = epsilon;
            else
                epsilon_t = 1e-6;
            end
            
            r = abs(v1 - v2) < epsilon_t;
            o = all(r(:));
        end
        
        function [tiles_image] = format_as_tiles(images,tiles_row_count,tiles_col_count,bypass_unitreal)
            assert(tc.tensor(images,4) && ...
                   (tc.unitreal(images) || (exist('bypass_unitreal','var') && tc.number(images))));
            assert(~exist('tiles_row_count','var') || ...
                   (tc.scalar(tiles_row_count) && tc.natural(tiles_row_count) && (tiles_row_count > 1)));
            assert(~exist('tiles_col_count','var') || ...
                   (tc.scalar(tiles_col_count) && tc.natural(tiles_col_count) && (tiles_col_count > 1)));
            assert(~(exist('tiles_row_count','var') && exist('tiles_col_count','var')) || ...
                     (tiles_row_count * tiles_col_count >= size(images,4)));
            assert(~(exist('tiles_row_count','var') && ~exist('tiles_col_count','var')) || ...
                     (tiles_row_count * tiles_row_count >= size(images,4)));
            assert(~exist('bypas_unitreal','var') || (tc.scalar(bypass_unitreal) && tc.logical(bypass_unitreal) && bypass_unitreal));
            
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
                for i = 1:tiles_row_count_t
                    for j = 1:tiles_col_count_t
                        if (i - 1) * tiles_col_count_t + j <= images_count
                            image = images(:,:,layer,(i - 1) * tiles_col_count_t + j);
                            tiles_image(((i - 1)*(row_count + 1) + 2):(i*(row_count + 1)),...
                                        ((j - 1)*(col_count + 1) + 2):(j*(col_count + 1)),layer) = image;
                        else
                            tiles_image(((i - 1)*(row_count + 1) + 2):(i*(row_count + 1)),...
                                        ((j - 1)*(col_count + 1) + 2):(j*(col_count + 1)),layer) = 0.25 * ones(row_count,col_count);
                        end
                    end
                end
            end
        end
        
        function [new_images] = clamp_images_to_unit(images)
            assert(tc.tensor(images,4) && tc.number(images));
            
            new_images = max(min(images,1),0);
        end
        
        function [new_images] = remap_images_to_unit(images,mode)
            assert(tc.tensor(images,4) && tc.number(images));
            assert(~exist('mode','var') || ...
                   (tc.scalar(mode) && tc.string(mode) && ...
                     (strcmp(mode,'local') || strcmp(mode,'global'))));
                 
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
                    for i = 1:images_count
                        im_min = min(min(images(:,:,layer,i)));
                        im_max = max(max(images(:,:,layer,i)));
                    
                        new_images(:,:,layer,i) = (images(:,:,layer,i) - im_min) / (im_max - im_min);
                    end
                end
            else
                im_min = min(images(:));
                im_max = max(images(:));
                
                new_images = (images - im_min) / (im_max - im_min);
            end
        end
        
        function [o] = same_classes(classes1,classes2)
            assert(tc.vector(classes1) && tc.labels(classes1));
            assert(tc.vector(classes2) && tc.labels(classes2));
            
            o = true;
            o = o && tc.match_dims(classes1,classes2);
            o = o && ((tc.logical(classes1) && tc.logical(classes2)) || ...
                      (tc.natural(classes1) && tc.natural(classes2)) || ...
                      (tc.cell(classes1) && tc.cell(classes2)));
            
            if tc.cell(classes1)
                o = o && tc.check(arrayfun(@(i)tc.check(classes1{i} == classes2{i}),1:length(classes1)));
            else
                o = o && tc.check(classes1 == classes2);
            end
        end
        
        function [params_list] = gen_all_params(params)
            assert(tc.scalar(params) && (isstruct(params)) && ...
                   tc.check(cellfun(@(c)tc.vector(c) && tc.number(c),struct2cell(params))));
            
            params_names = fieldnames(params);
            params_non_expand = cell(length(params_names),1);
            
            for i = 1:length(params_names)
                params_non_expand{i} = [params.(params_names{i})];
            end
            
            params_expand = feval(@allcomb,params_non_expand{:});
            
            special = cell(2 * length(params_names),1);
            
            for i = 1:length(params_names)
                special{2*(i - 1) + 1} = params_names{i};
                special{2*(i - 1) + 2} = num2cell(params_expand(:,i));
            end
            
            params_list = feval(@struct,special{:});
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
            
            fprintf('    With relaxed input ("images" does not have to be "unitreal").\n');
            
            t = 3*rand(20,20,1,36) - 1.5;
            tt = utils.format_as_tiles(t,6,6,true);
            
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
            
            t = 4 * rand(20,20,1,36) - 2;
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
            
            fprintf('    With color images.\n');
            
            t = rand(20,20,3,36) + 0.2;
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
            
            t = 4 * rand(20,20,1,36) - 2;
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
            
            t = 4 * rand(20,20,1,36) - 2;
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
            
            t = 4 * rand(20,20,1,36) - 2;
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
            
            fprintf('    With color images.\n');
            
            t = 4 * rand(20,20,3,36) - 2;
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
            
            fprintf('  Function "same_classes".\n');
            
            assert(utils.same_classes(true,true) == true);
            assert(utils.same_classes([1 2 3],[1 2 3]) == true);
            assert(utils.same_classes({'1' '2'},{'1' '2'}) == true);
            assert(utils.same_classes([1 2 3],[1 2]) == false);
            assert(utils.same_classes([1 2],[1 2 3 4]) == false);
            assert(utils.same_classes(true,1) == false);
            assert(utils.same_classes(false,{'1'}) == false);
            assert(utils.same_classes(1,{'1'}) == false);
            assert(utils.same_classes([1 2],[3 4]) == false);
            assert(utils.same_classes({'1' '2' '3'},{'1' '2' 'none'}) == false);
            
            fprintf('  Function "gen_all_params".\n');
            
            test_params.a = [1 2 3];
            test_params.b = [2 3];
            test_params.c = 4;
            
            params_list = utils.gen_all_params(test_params);
            
            assert(params_list(1).a == 1);
            assert(params_list(1).b == 2);
            assert(params_list(1).c == 4);
            assert(params_list(2).a == 1);
            assert(params_list(2).b == 3);
            assert(params_list(2).c == 4);
            assert(params_list(3).a == 2);
            assert(params_list(3).b == 2);
            assert(params_list(3).c == 4);
            assert(params_list(4).a == 2);
            assert(params_list(4).b == 3);
            assert(params_list(4).c == 4);
            assert(params_list(5).a == 3);
            assert(params_list(5).b == 2);
            assert(params_list(5).c == 4);
            assert(params_list(6).a == 3);
            assert(params_list(6).b == 3);
            assert(params_list(6).c == 4);
            
            clearvars -except display;
        end
    end
end
