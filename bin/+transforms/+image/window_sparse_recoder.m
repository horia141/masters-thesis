classdef window_sparse_recoder < transform
    properties (GetAccess=public,SetAccess=immutable)
        pooled_patch_row_count;
        pooled_patch_col_count;
        patch_count_row;
        patch_count_col;
        t_zca;
        t_dictionary;
        dictionary_ctor_fn;
        nonlinear_fn;
        reduce_fn;
        features_mult_factor;
        initial_value;
        patches_count;
        patch_row_count;
        patch_col_count;
        patch_required_variance;
        dictionary_type;
        dictionary_params;
        window_step;
        nonlinear_type;
        nonlinear_params;
        reduce_type;
        reduce_spread;
    end
    
    methods (Access=public)
        function [obj] = window_sparse_recoder(train_sample_plain,patches_count,patch_row_count,patch_col_count,patch_required_variance,...
                                                                  dictionary_type,dictionary_params,window_step,nonlinear_type,nonlinear_params,reduce_type,reduce_spread,logger)
            assert(tc.dataset_image(train_sample_plain));
            assert(size(train_sample_plain,3) == 1);
            assert(tc.scalar(patches_count));
            assert(tc.scalar(patches_count));
            assert(tc.natural(patches_count));
            assert(patches_count >= 1);
            assert(tc.scalar(patch_row_count));
            assert(tc.natural(patch_row_count));
            assert(patch_row_count >= 1);
            assert(mod(patch_row_count,2) == 1);
            assert(tc.scalar(patch_col_count));
            assert(tc.natural(patch_col_count));
            assert(patch_col_count >= 1);
            assert(mod(patch_col_count,2) == 1);
            assert(tc.scalar(patch_required_variance));
            assert(tc.number(patch_required_variance));
            assert(patch_required_variance >= 0);
            assert(tc.scalar(dictionary_type));
            assert(tc.string(dictionary_type));
            assert(tc.one_of(dictionary_type,'Dict','Learn:Grad','Random:Filters','Random:Instances'));
            assert(tc.empty(dictionary_params) || tc.vector(dictionary_params));
            assert(tc.empty(dictionary_params) || tc.cell(dictionary_params));
            assert(tc.scalar(window_step));
            assert(tc.natural(window_step));
            assert(window_step >= 1);
            assert(tc.scalar(nonlinear_type));
            assert(tc.string(nonlinear_type));
            assert(tc.one_of(nonlinear_type,'Linear','Logistic'));
            assert(tc.empty(nonlinear_params) || tc.vector(nonlinear_params));
            assert(tc.empty(nonlinear_params) || tc.cell(nonlinear_params));
            assert(tc.scalar(reduce_type));
            assert(tc.string(reduce_type));
            assert(tc.one_of(reduce_type,'Subsample','Sqr','Max','MinMax'));
            assert(tc.scalar(reduce_spread));
            assert(tc.natural(reduce_spread));
            assert(reduce_spread >= 1);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(mod(size(train_sample_plain,1) - 1,window_step) == 0); % A BIT OF A HACK
            assert(mod(size(train_sample_plain,2) - 1,window_step) == 0); % A BIT OF A HACK
            assert(mod((size(train_sample_plain,1) - 1) / window_step + 1,reduce_spread) == 0);
            assert(mod((size(train_sample_plain,2) - 1) / window_step + 1,reduce_spread) == 0);
            
            [d,dr,dc,~] = dataset.geometry(train_sample_plain);
            
            patch_count_row_t = (dr - 1) / window_step + 1;
            patch_count_col_t = (dc - 1) / window_step + 1;
            pooled_patch_row_count_t = patch_count_row_t / reduce_spread;
            pooled_patch_col_count_t = patch_count_col_t / reduce_spread;
            
            if tc.same(dictionary_type,'Dict')
                dictionary_ctor_fn_t = @transforms.record.dictionary;
            elseif tc.same(dictionary_type,'Learn:Grad')
                dictionary_ctor_fn_t = @transforms.record.dictionary.learn.grad;
            elseif tc.same(dictionary_type,'Random:Filters')
                dictionary_ctor_fn_t = @transforms.record.dictionary.random.filters;
            else
                dictionary_ctor_fn_t = @transforms.record.dictionary.random.instances;
            end
            
            if tc.same(nonlinear_type,'Linear')
                nonlinear_fn_t = @transforms.image.window_sparse_recoder.linear;
            else
                nonlinear_fn_t = @transforms.image.window_sparse_recoder.logistic;
            end

            if tc.same(reduce_type,'Subsample')
                reduce_fn_t = @transforms.image.window_sparse_recoder.subsample;
                features_mult_factor_t = 1;
                initial_value_t = 0;
            elseif tc.same(reduce_type,'Sqr')
                reduce_fn_t = @transforms.image.window_sparse_recoder.sqr;
                features_mult_factor_t = 1;
                initial_value_t = 0;
            elseif tc.same(reduce_type,'Max')
                reduce_fn_t = @transforms.image.window_sparse_recoder.max;
                features_mult_factor_t = 1;
                initial_value_t = 0;
            else
                reduce_fn_t = @transforms.image.window_sparse_recoder.minmax;
                features_mult_factor_t = 2;
                initial_value_t = NaN;
            end
            
            t_patches = transforms.image.patch_extract(train_sample_plain,patches_count,patch_row_count,patch_col_count,patch_required_variance,logger.new_transform('Building patch extract transform'));
            patches_1 = t_patches.code(train_sample_plain,logger.new_transform('Extracting patches'));
            patches_2 = dataset.flatten_image(patches_1);
            patches_3 = bsxfun(@minus,patches_2,mean(patches_2,1));
%             t_zca_t = transforms.record.zca(patches_3,logger.new_transform('Building ZCA transform'));
%             patches_4 = t_zca_t.code(patches_3,logger.new_transform('Applying ZCA transform'));
            patches_4 = patches_3;
            t_dictionary_t = dictionary_ctor_fn_t(patches_4,dictionary_params{:},logger.new_transform('Building patches dictionary'));
            
            input_geometry = [d,dr,dc,1];
            output_geometry = features_mult_factor_t * pooled_patch_row_count_t * pooled_patch_col_count_t * t_dictionary_t.word_count;
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.pooled_patch_row_count = pooled_patch_row_count_t;
            obj.pooled_patch_col_count = pooled_patch_col_count_t;
            obj.patch_count_row = patch_count_row_t;
            obj.patch_count_col = patch_count_col_t;
            obj.t_zca = 0; %t_zca_t;
            obj.t_dictionary = t_dictionary_t;
            obj.dictionary_ctor_fn = dictionary_ctor_fn_t;
            obj.nonlinear_fn = nonlinear_fn_t;
            obj.reduce_fn = reduce_fn_t;
            obj.features_mult_factor = features_mult_factor_t;
            obj.initial_value = initial_value_t;
            obj.patches_count = patches_count;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.patch_required_variance = patch_required_variance;
            obj.dictionary_type = dictionary_type;
            obj.dictionary_params = dictionary_params;
            obj.window_step = window_step;
            obj.nonlinear_type = nonlinear_type;
            obj.nonlinear_params = nonlinear_params;
            obj.reduce_type = reduce_type;
            obj.reduce_spread = reduce_spread;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            N = dataset.count(sample_plain);
            [~,dr,dc,~] = dataset.geometry(sample_plain);
            
            sample_plain_padded = zeros(dr + obj.patch_row_count - 1,dc + obj.patch_col_count - 1,1,N);
            sample_plain_padded(((obj.patch_row_count - 1) / 2 + 1):(end - (obj.patch_row_count - 1)/2),((obj.patch_col_count - 1) / 2 + 1):(end - (obj.patch_col_count - 1)/2),:,:) = sample_plain;
            
            full_sample = zeros(obj.features_mult_factor * obj.t_dictionary.word_count,N,obj.pooled_patch_row_count,obj.pooled_patch_col_count);
            local_sample_plain = shiftdim(sample_plain_padded,2);
            
            current_patch = 1;
            total_patches_count = obj.patch_count_row * obj.patch_count_col;
            
            for ii = 1:obj.pooled_patch_row_count
                for jj = 1:obj.pooled_patch_col_count
                    logger.beg_node('Output %02d%02d',ii,jj);

                    source_patches_row = ((ii - 1)*obj.reduce_spread + 1):(ii * obj.reduce_spread);
                    source_patches_col = ((jj - 1)*obj.reduce_spread + 1):(jj * obj.reduce_spread);
                    
                    local_sample = obj.initial_value * ones(obj.features_mult_factor * obj.t_dictionary.word_count,N);
           
                    for ii_1 = source_patches_row
                        for jj_1 = source_patches_col
                            logger.beg_node('Patch %02dx%02d %02dx%02d (%.2f/100)',((ii_1 - 1) * obj.window_step + 1),((jj_1 - 1) * obj.window_step + 1),...
                                                                                   ((ii_1 - 1) * obj.window_step + obj.patch_row_count),((jj_1 - 1) * obj.window_step + obj.patch_col_count),...
                                                                                    100 * current_patch / total_patches_count);

                            % Should check again that what we do here actually works.
                            local_sample_1 = local_sample_plain(:,:,((ii_1 - 1) * obj.window_step + 1):((ii_1 - 1) * obj.window_step + obj.patch_row_count),...
                                                                    ((jj_1 - 1) * obj.window_step + 1):((jj_1 - 1) * obj.window_step + obj.patch_col_count));
                            local_sample_2 = reshape(local_sample_1,N,obj.patch_row_count * obj.patch_col_count)'; 
                            local_sample_3 = bsxfun(@minus,local_sample_2,mean(local_sample_2,1));
%                             local_sample_4 = obj.t_zca.code(local_sample_3,logger);
                            local_sample_4 = local_sample_3;
                            local_sample_5 = obj.t_dictionary.code(local_sample_4,logger);
                            local_sample_6 = obj.nonlinear_fn(local_sample_5,obj.nonlinear_params{:});

                            local_sample = obj.reduce_fn(local_sample,local_sample_6);

                            logger.end_node();
                            
                            current_patch = current_patch + 1;
                        end
                    end

                    full_sample(:,:,ii,jj) = local_sample;

                    logger.end_node();
                end
            end
            
            sample_coded = reshape(shiftdim(full_sample,2),obj.output_geometry,N);
        end
    end
    
    methods (Static,Access=protected)
        function [o] = linear(i)
            o = i;
        end
        
        function [o] = logistic(i)
            o = 1 ./ (1 + 2.71828183.^ (-i)) - 0.5;
        end

        function [o] = subsample(acc,A)
            if tc.same(acc,zeros(size(acc)))
                o = A;
            else
                o = acc;
            end
        end
        
        function [o] = sqr(acc,A)
            o = acc + A .^ 2;
        end
        
        function [o] = max(acc,A)
            o = max(acc,abs(A));
        end
        
        function [o] = minmax(acc,A)
            o_1 = [max(acc(1:2:end,:),A);min(acc(2:2:end,:),A)];
            o = [o_1(1:2:end,:);o_1(2:2:end,:)];
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "transforms.image.window_sparse_recoder".\n');
        end
    end
end
