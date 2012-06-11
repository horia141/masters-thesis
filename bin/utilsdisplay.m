classdef utilsdisplay
    methods (Static,Access=public)
        function [] = sparse_basis(dict,row_count,col_count)
            assert(tc.matrix(dict));
            assert(tc.scalar(row_count));
            assert(tc.natural(row_count));
            assert(row_count >= 1);
            assert(tc.scalar(col_count));
            assert(tc.natural(col_count));
            assert(col_count >= 1);
            assert(row_count * col_count == size(dict,2));
            
            im_dict = zeros(row_count,col_count,1,size(dict,1));
            
            for ii = 1:size(dict,1)
                im_dict(:,:,1,ii) = reshape(dict(ii,:),row_count,col_count);
            end
            
            imshow(utils.format_as_tiles(utils.remap_images_to_unit(im_dict,'global')));
        end
        
        function [] = window_coded_output(sample_plain,sample_coded,dict_count,pooled_patch_row_count,pooled_patch_col_count,tiles_row_count,tiles_col_count)
            assert(tc.dataset_image(sample_plain));
            assert(tc.dataset_record(sample_coded));
            assert(dataset.count(sample_plain) == dataset.count(sample_coded));
            assert(tc.scalar(dict_count));
            assert(tc.natural(dict_count));
            assert(dict_count >= 1);
            assert(tc.scalar(pooled_patch_row_count));
            assert(tc.natural(pooled_patch_row_count));
            assert(pooled_patch_row_count >= 1);
            assert(tc.scalar(pooled_patch_col_count));
            assert(tc.natural(pooled_patch_col_count));
            assert(pooled_patch_col_count >= 1);
            assert(dataset.geometry(sample_coded) == dict_count * pooled_patch_row_count * pooled_patch_col_count);
            assert(~exist('tiles_row_count','var') || tc.scalar(tiles_row_count));
            assert(~exist('tiles_row_count','var') || tc.natural(tiles_row_count));
            assert(~exist('tiles_row_count','var') || (tiles_row_count >= 1));
            assert(~exist('tiles_col_count','var') || tc.scalar(tiles_col_count));
            assert(~exist('tiles_col_count','var') || tc.natural(tiles_col_count));
            assert(~exist('tiles_col_count','var') || (tiles_col_count >= 1));
            assert(~(exist('tiles_row_count','var') && exist('tiles_col_count','var')) || ...
                     (tiles_row_count * tiles_col_count >= dict_count));
            assert(~(exist('tiles_row_count','var') && ~exist('tiles_col_count','var')) || ...
                     (tiles_row_count * tiles_row_count >= dict_count));
                 
            if exist('tiles_row_count','var') && exist('tiles_col_count','var')
                tiles_row_count_t = tiles_row_count;
                tiles_col_count_t = tiles_col_count;
            elseif exist('tiles_row_count','var') && ~exist('tiles_col_count','var')
                tiles_row_count_t = tiles_row_count;
                tiles_col_count_t = tiles_row_count;
            else
                tiles_row_count_t = ceil(sqrt(dict_count));
                tiles_col_count_t = ceil(sqrt(dict_count));
            end
            
            N = dataset.count(sample_plain);
            sample_reshaped_t1 = reshape(sample_coded,pooled_patch_row_count,pooled_patch_col_count,1,dict_count*N);
            sample_reshaped_t2 = utils.remap_images_to_unit(sample_reshaped_t1,'global');
            sample_reshaped = reshape(sample_reshaped_t2,pooled_patch_row_count,pooled_patch_col_count,1,dict_count,N);
            
            for ii = 1:N
                subplot(2,1,1);
                imshow(sample_plain(:,:,:,ii));
                subplot(2,1,2);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(sample_reshaped(:,:,:,:,ii),'global'),tiles_row_count_t,tiles_col_count_t));
                pause;
            end
        end        
    end
end
