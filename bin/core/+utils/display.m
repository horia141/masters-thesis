classdef display
    methods (Static,Access=public)
        function [] = as_tiles(images,tiles_format)
            assert(check.dataset_image(images));
            assert(~exist('tiles_format','var') || (check.scalar(tiles_format) || (check.vector(tiles_format) && length(tiles_format) == 2)));
            assert(~exist('tiles_format','var') || (check.natural(tiles_format)));
            assert(~exist('tiles_format','var') || (check.checkv(tiles_format >= 1)));
            assert(~exist('tiles_format','var') || (~(length(tiles_format) == 2) || prod(tiles_format) >= dataset.count(images)));
            
            N = dataset.count(images);
            [~,dr,dc,dl] = dataset.geometry(images);
            
            if exist('tiles_format','var')
                if check.scalar(tiles_format)
                    tiles_row_count = tiles_format;
                    tiles_col_count = ceil(N / tiles_format);
                else
                    tiles_row_count = tiles_format(1);
                    tiles_col_count = tiles_format(2);
                end
            else
                tiles_row_count = ceil(sqrt(N));
                tiles_col_count = ceil(sqrt(N));
            end
            
            tiles = ones((dr + 1) * tiles_row_count + 1,(dc + 1) * tiles_col_count + 1,dl);
            
            for ii = 1:tiles_row_count
                for jj = 1:tiles_col_count
                    if (ii - 1) * tiles_col_count + jj <= N
                        image = images(:,:,:,(ii - 1) * tiles_col_count + jj);
                        tiles(((ii - 1) * (dr + 1) + 2):(ii * (dr + 1)),...
                              ((jj - 1) * (dc + 1) + 2):(jj * (dc + 1)),:) = image;
                    else
                        tiles(((ii - 1) * (dr + 1) + 2):(ii * (dr + 1)),...
                              ((jj - 1) * (dc + 1) + 2):(jj * (dc + 1)),:) = 0.25;
                    end
                end
            end
            
            imshow(tiles);
        end

        function [] = dictionary(dict,row_count,col_count,tiles_format)
            assert(check.matrix(dict));
            assert(check.number(dict));
            assert(check.scalar(row_count));
            assert(check.natural(row_count));
            assert(row_count >= 1);
            assert(check.scalar(col_count));
            assert(check.natural(col_count));
            assert(col_count >= 1);
            assert(~exist('tiles_format','var') || (check.scalar(tiles_format) || (check.vector(tiles_format) && length(tiles_format) == 2)));
            assert(~exist('tiles_format','var') || (check.natural(tiles_format)));
            assert(~exist('tiles_format','var') || (check.checkv(tiles_format >= 1)));
            assert(row_count * col_count == size(dict,2));
            assert(~exist('tiles_format','var') || (~(length(tiles_format) == 2) || prod(tiles_format) >= size(dict,1)));
            
            if exist('tiles_format','var')
                tiles_format_t = {tiles_format};
            else
                tiles_format_t = {};
            end

            im_dict = zeros(row_count,col_count,1,size(dict,1));
            
            for ii = 1:size(dict,1)
                im_dict(:,:,1,ii) = reshape(dict(ii,:),row_count,col_count);
            end

            utils.display.as_tiles(utils.common.remap_images_to_unit(im_dict),tiles_format_t{:});
        end
        
        function [] = coded_output(sample_plain,sample_coded,word_count,do_polarity_split,pooled_patch_row_count,pooled_patch_col_count,new_frame_wait_time,tiles_format)
            assert(check.dataset_image(sample_plain));
            assert(check.dataset_record(sample_coded));
            assert(check.scalar(word_count));
            assert(check.natural(word_count));
            assert(word_count >= 1);
            assert(check.scalar(do_polarity_split));
            assert(check.logical(do_polarity_split));
            assert(check.scalar(pooled_patch_row_count));
            assert(check.natural(pooled_patch_row_count));
            assert(pooled_patch_row_count >= 1);
            assert(check.scalar(pooled_patch_col_count));
            assert(check.natural(pooled_patch_col_count));
            assert(pooled_patch_col_count >= 1);
            assert(~exist('new_frame_wait_time','var') || check.scalar(new_frame_wait_time));
            assert(~exist('new_frame_wait_time','var') || ((check.number(new_frame_wait_time) && (new_frame_wait_time > 0)) || (new_frame_wait_time == -1)));
            assert(~exist('tiles_format','var') || (check.scalar(tiles_format) || (check.vector(tiles_format) && length(tiles_format) == 2)));
            assert(~exist('tiles_format','var') || (check.natural(tiles_format)));
            assert(~exist('tiles_format','var') || (check.checkv(tiles_format >= 1)));
            assert(dataset.count(sample_plain) == dataset.count(sample_coded));
            assert((do_polarity_split && (dataset.geometry(sample_coded) == 2 * word_count * pooled_patch_row_count * pooled_patch_col_count)) || ...
	    	       (~do_polarity_split && (dataset.geometry(sample_coded) == 1 * word_count * pooled_patch_row_count * pooled_patch_col_count)))
            assert(~exist('tiles_format','var') || (~(length(tiles_format) == 2) || prod(tiles_format) >= word_count));
            
            if exist('tiles_format','var')
                tiles_format_t = {tiles_format};
            else
                tiles_format_t = {};
            end

            if do_polarity_split
                reshape_mult_factor = 2;
            else
                reshape_mult_factor = 1;
            end
            
            N = dataset.count(sample_plain);
            sample_coded_full = full(sample_coded);
            sample_reshaped_t1 = reshape(sample_coded_full,pooled_patch_row_count,pooled_patch_col_count,1,reshape_mult_factor*word_count*N);
            sample_reshaped_t2 = utils.common.remap_images_to_unit(sample_reshaped_t1);
            sample_reshaped = reshape(sample_reshaped_t2,pooled_patch_row_count,pooled_patch_col_count,reshape_mult_factor,word_count,N);
            
            if do_polarity_split
                for ii = 1:N
                    subplot(3,1,1);
                    imshow(sample_plain(:,:,:,ii));
                    subplot(3,1,2);
                    utils.display.as_tiles(sample_reshaped(:,:,1,:,ii),tiles_format_t{:});
                    subplot(3,1,3);
                    utils.display.as_tiles(sample_reshaped(:,:,2,:,ii),tiles_format_t{:});
                    if new_frame_wait_time == -1
                        pause;
                    else
                        pause(new_frame_wait_time);
                    end
                end
            else
                for ii = 1:N
                    subplot(2,1,1);
                    imshow(sample_plain(:,:,:,ii));
                    subplot(2,1,2);
                    utils.display.as_tiles(sample_reshaped(:,:,:,:,ii),tiles_format_t{:});
                    if new_frame_wait_time == -1
                        pause;
                    else
                        pause(new_frame_wait_time);
                    end
                end
            end
        end

        function [] = classification_border(cl,sample_tr,sample_ts,ci_tr,ci_ts,range)
            assert(check.scalar(cl));
            assert(check.classifier(cl));
            assert(check.dataset_record(sample_tr));
            assert(check.same(dataset.geometry(sample_tr),2));
            assert(check.dataset_record(sample_ts));
            assert(check.same(dataset.geometry(sample_ts),2));
            assert(check.scalar(ci_tr));
            assert(check.classifier_info(ci_tr));
            assert(check.scalar(ci_ts));
            assert(check.classifier_info(ci_ts));
            assert(check.vector(range));
            assert(length(range) == 4);
            assert(check.number(range));
            assert(range(1) < range(2));
            assert(range(3) < range(4));
            assert(ci_tr.compatible(sample_tr));
            assert(ci_ts.compatible(sample_ts));
            
            clf(gcf());
            subplot(2,2,1);
            hold on;
            gscatter(sample_tr(1,:),sample_tr(2,:),ci_tr.labels_idx,'rgb','o',6);
            gscatter(sample_ts(1,:),sample_ts(2,:),ci_ts.labels_idx,'rgb','o',6);
            [ptmp_x,ptmp_y] = meshgrid(-1:0.05:5,-1:0.05:5);
            ptmp = sparse([ptmp_x(:),ptmp_y(:)]');
            [ptmp2_x,ptmp2_y] = meshgrid(-1:0.2:5,-1:0.2:5);
            ptmp2 = sparse([ptmp2_x(:),ptmp2_y(:)]');
            l = cl.classify(ptmp,-1);
            [~,cfd] = cl.classify(ptmp2,-1);
            gscatter(ptmp(1,:),ptmp(2,:),l,'rgb','*',2);
            hold off;
            axis(range);
            subplot(2,2,2);
            mesh(-1:0.2:5,-1:0.2:5,reshape(cfd(1,:),31,31),cat(3,reshape(cfd(1,:),31,31),zeros(31,31),zeros(31,31)));
            axis([range 0 1]);
            subplot(2,2,3);
            mesh(-1:0.2:5,-1:0.2:5,reshape(cfd(2,:),31,31),cat(3,zeros(31,31),reshape(cfd(2,:),31,31),zeros(31,31)));
            axis([range 0 1]);
            subplot(2,2,4);
            if size(cfd,1) > 2
                mesh(-1:0.2:5,-1:0.2:5,reshape(cfd(3,:),31,31),cat(3,zeros(31,31),zeros(31,31),reshape(cfd(3,:),31,31)));
                axis([range 0 1]);
            end
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "utils.display".\n');
            
            fprintf('  Function "as_tiles".\n');
            
            fprintf('    With implicit "tiles_format".\n');
            
            t = dataset.load('../../test/scenes_small.mat');
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,1,1);
                utils.display.as_tiles(t);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    With one value for "tiles_format".\n');
            
            t = dataset.load('../../test/scenes_small.mat');
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,1,1);
                utils.display.as_tiles(t,1);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    With two values for "tiles_format".\n');
            
            t = dataset.load('../../test/scenes_small.mat');
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,1,1);
                utils.display.as_tiles(t,[3 4]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('  Function "dictionary".\n');
            
            fprintf('    With implicit "tiles_format".\n');
            
            vvv = load('../../test/sparse_dictionary.mat');
            t = vvv.saved_dict_11x11_144;
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,1,1);
                utils.display.dictionary(t,11,11);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    With one value for "tiles_format".\n');
            
            vvv = load('../../test/sparse_dictionary.mat');
            t = vvv.saved_dict_11x11_144;
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,1,1);
                utils.display.dictionary(t,11,11,10);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    With two values for "tiles_format".\n');
            
            vvv = load('../../test/sparse_dictionary.mat');
            t = vvv.saved_dict_11x11_144;
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,1,1);
                utils.display.dictionary(t,11,11,[20 10]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('  Function "coded_output".\n');
            
            fprintf('    NOT YET TESTED!!!\n');
            
            clearvars -except test_figure;
            
            fprintf('  Function "classification_border".\n');
            
            fprintf('    NOT YET TESTED!!!\n');
            
            clearvars -except test_figure;
        end
    end
end
