classdef window_sparse_recoder < transform
    properties (GetAccess=public,SetAccess=immutable)
        pooled_patch_row_count;
        pooled_patch_col_count;
        patch_count_row;
        patch_count_col;
        t_zca;
        t_sparse;
        patches_count;
        patch_row_count;
        patch_col_count;
        patch_required_variance;
        coding_fn;
        word_count;
        coeffs_count;
        initial_learning_rate;
        final_learing_rate;
        window_step;
        reduce_fn;
        reduce_spread;
    end
    
    methods (Access=public)
        function [obj] = window_sparse_recoder(train_sample_plain,patches_count,patch_row_count,patch_col_count,patch_required_variance,coding_fn,word_count,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count,window_step,reduce_fn,reduce_spread,logger)
            assert(tc.dataset_image(train_sample_plain));
            assert(size(train_sample_plain,3) == 1);
            assert(tc.scalar(patches_count));
            assert(tc.scalar(patches_count));
            assert(tc.natural(patches_count));
            assert(patches_count >= 1);
            assert(tc.scalar(patch_row_count));
            assert(tc.natural(patch_row_count));
            assert(patch_row_count >= 1);
            assert(tc.scalar(patch_col_count));
            assert(tc.natural(patch_col_count));
            assert(patch_col_count >= 1);
            assert(tc.scalar(patch_required_variance));
            assert(tc.number(patch_required_variance));
            assert(patch_required_variance >= 0);
            assert(tc.scalar(coding_fn));
            assert(tc.function_h(coding_fn));
            assert(tc.one_of(coding_fn,@transforms.sparse.gdmp.matching_pursuit,@transforms.sparse.gdmp.ortho_matching_pursuit));
            assert(tc.scalar(word_count));
            assert(tc.natural(word_count));
            assert(word_count >= 1);
            assert(tc.scalar(coeffs_count));
            assert(tc.natural(coeffs_count));
            assert(coeffs_count >= 1 && coeffs_count <= word_count);
            assert(tc.scalar(initial_learning_rate));
            assert(tc.number(initial_learning_rate));
            assert(initial_learning_rate > 0);
            assert(tc.scalar(final_learning_rate));
            assert(tc.number(final_learning_rate));
            assert(final_learning_rate > 0);
            assert(final_learning_rate <= initial_learning_rate);
            assert(tc.scalar(max_iter_count));
            assert(tc.natural(max_iter_count));
            assert(max_iter_count >= 1);
            assert(tc.scalar(window_step));
            assert(tc.natural(window_step));
            assert(window_step >= 1);
            assert(tc.scalar(reduce_fn));
            assert(tc.function_h(reduce_fn));
            assert(tc.one_of(reduce_fn,@transforms.image.window_sparse_recoder.sqr,@transforms.image.window_sparse_recoder.max));
            assert(tc.scalar(reduce_spread));
            assert(tc.natural(reduce_spread));
            assert(reduce_spread >= 1);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(mod(size(train_sample_plain,1) - patch_row_count,window_step) == 0); % A BIT OF A HACK
            assert(mod(size(train_sample_plain,2) - patch_col_count,window_step) == 0);
            assert(mod((size(train_sample_plain,1) - patch_row_count) / window_step + 1,reduce_spread) == 0);
            assert(mod((size(train_sample_plain,2) - patch_col_count) / window_step + 1,reduce_spread) == 0);
            
            [d,dr,dc,~] = dataset.geometry(train_sample_plain);
            
            patch_count_row_t = (dr - patch_row_count) / window_step + 1;
            patch_count_col_t = (dc - patch_col_count) / window_step + 1;
            pooled_patch_row_count_t = patch_count_row_t / reduce_spread;
            pooled_patch_col_count_t = patch_count_col_t / reduce_spread;
            
            t_patches = transforms.image.patch_extract(train_sample_plain,patches_count,patch_row_count,patch_col_count,patch_required_variance,logger.new_transform('Building patch extract transform'));
            patches_1 = t_patches.code(train_sample_plain,logger.new_transform('Extracting patches'));
            patches_2 = dataset.flatten_image(patches_1);
            patches_3 = bsxfun(@minus,patches_2,mean(patches_2,1));
            t_zca_t = transforms.record.zca(patches_3,logger.new_transform('Building ZCA transform'));
            patches_4 = t_zca_t.code(patches_3,logger.new_transform('Applying ZCA transform'));
            t_sparse_t = transforms.sparse.gdmp(patches_4,coding_fn,word_count,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count,logger.new_transform('Building sparse coder'));
            
            input_geometry = [d,dr,dc,1];
            output_geometry = pooled_patch_row_count_t * pooled_patch_col_count_t * 2 * word_count;
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.pooled_patch_row_count = pooled_patch_row_count_t;
            obj.pooled_patch_col_count = pooled_patch_col_count_t;
            obj.patch_count_row = patch_count_row_t;
            obj.patch_count_col = patch_count_col_t;
            obj.t_zca = t_zca_t;
            obj.t_sparse = t_sparse_t;
            obj.patches_count = patches_count;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.patch_required_variance = patch_required_variance;
            obj.coding_fn = coding_fn;
            obj.word_count = word_count;
            obj.coeffs_count = coeffs_count;
            obj.initial_learning_rate = initial_learning_rate;
            obj.final_learing_rate = final_learning_rate;
            obj.window_step = window_step;
            obj.reduce_fn = reduce_fn;
            obj.reduce_spread = reduce_spread;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            N = dataset.count(sample_plain);
            
            full_sample = zeros(2*obj.word_count,N,obj.pooled_patch_row_count,obj.pooled_patch_col_count);
            local_sample = sparse([],[],[],2*obj.word_count,N,0);
            local_sample_plain = shiftdim(sample_plain,2);
	    % draw_sample = zeros(2*obj.word_count,N,obj.patch_row_count,obj.patch_col_count);
            
            for ii = 1:obj.pooled_patch_row_count
                for jj = 1:obj.pooled_patch_col_count
                    logger.beg_node('Output %02d%02d',ii,jj);

                    source_patches_row = ((ii - 1)*obj.reduce_spread + 1):(ii * obj.reduce_spread);
                    source_patches_col = ((jj - 1)*obj.reduce_spread + 1):(jj * obj.reduce_spread);
                    
                    for ii_1 = source_patches_row
                        for jj_1 = source_patches_col
                            logger.beg_node('Patch %02dx%02d %02dx%02d',ii_1,jj_1,ii_1 + obj.reduce_spread,jj_1 + obj.reduce_spread);

                            % Should check again that what we do here actually works.
                            local_sample_1 = local_sample_plain(:,:,((ii_1 - 1) * obj.window_step + 1):((ii_1 - 1) * obj.window_step + obj.patch_row_count),...
                                                                    ((jj_1 - 1) * obj.window_step + 1):((jj_1 - 1) * obj.window_step + obj.patch_col_count));
                            local_sample_2 = reshape(local_sample_1,N,obj.patch_row_count * obj.patch_col_count)'; 
                            local_sample_3 = bsxfun(@minus,local_sample_2,mean(local_sample_2,1));
                            local_sample_4 = obj.t_zca.code(local_sample_3,logger);
                            local_sample_5 = obj.t_sparse.code(local_sample_4,logger);
                             
                            local_sample = obj.reduce_fn(local_sample,local_sample_5);
			  %  draw_sample(:,:,ii_1,jj_1) = local_sample_5;
                            
                            logger.end_node();
                        end
                    end

		            full_sample(:,:,ii,jj) = local_sample;

                    logger.end_node();
                end
            end
            
            sample_coded = reshape(shiftdim(full_sample,2),obj.output_geometry,N);
        end
    end
    
    methods (Static,Access=public)
        function [o] = sqr(acc,A)
            o = acc + A .^ 2;
        end
        
        function [o] = max(acc,A)
            o = max(acc,abs(A));
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "transforms.image.window_sparse_recoder".\n');
        end
    end
end
