classdef recoder < transform
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
        initial_value;
        patches_count;
        patch_row_count;
        patch_col_count;
        patch_required_variance;
        do_patch_zca;
        do_polarity_split;
        dictionary_type;
        dictionary_params;
        window_step;
        nonlinear_type;
        nonlinear_params;
        reduce_type;
        reduce_spread;
    end
    
    methods (Access=public)
        function [obj] = recoder(train_sample_plain,patches_count,patch_row_count,patch_col_count,patch_required_variance,do_patch_zca,do_polarity_split,...
                                                    dictionary_type,dictionary_params,window_step,nonlinear_type,nonlinear_params,reduce_type,reduce_spread,logger)
            assert(check.dataset_image(train_sample_plain));
            assert(size(train_sample_plain,3) == 1); % A BIT OF A HACK
            assert(check.scalar(patches_count));
            assert(check.scalar(patches_count));
            assert(check.natural(patches_count));
            assert(patches_count >= 1);
            assert(check.scalar(patch_row_count));
            assert(check.natural(patch_row_count));
            assert(patch_row_count >= 1);
            assert(mod(patch_row_count,2) == 1);
            assert(check.scalar(patch_col_count));
            assert(check.natural(patch_col_count));
            assert(patch_col_count >= 1);
            assert(mod(patch_col_count,2) == 1);
            assert(check.scalar(patch_required_variance));
            assert(check.number(patch_required_variance));
            assert(patch_required_variance >= 0);
            assert(check.scalar(do_patch_zca));
            assert(check.logical(do_patch_zca));
            assert(check.scalar(do_polarity_split));
            assert(check.logical(do_polarity_split));
            assert(check.scalar(dictionary_type));
            assert(check.string(dictionary_type));
            assert(check.one_of(dictionary_type,'Dict','Random:Filters','Random:Instances','Learn:Grad','Learn:GradSt'));
            assert(check.empty(dictionary_params) || check.vector(dictionary_params));
            assert(check.empty(dictionary_params) || check.cell(dictionary_params));
            assert(check.scalar(window_step));
            assert(check.natural(window_step));
            assert(window_step >= 1);
            assert(check.scalar(nonlinear_type));
            assert(check.string(nonlinear_type));
            assert(check.one_of(nonlinear_type,'Linear','Logistic'));
            assert(check.empty(nonlinear_params) || check.vector(nonlinear_params));
            assert(check.empty(nonlinear_params) || check.cell(nonlinear_params));
            assert(check.scalar(reduce_type));
            assert(check.string(reduce_type));
            assert(check.one_of(reduce_type,'Subsample','Sqr','Max','MaxMagnitude'));
            assert(check.scalar(reduce_spread));
            assert(check.natural(reduce_spread));
            assert(reduce_spread >= 1);
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
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
            
            if check.same(dictionary_type,'Dict')
                dictionary_ctor_fn_t = @transforms.record.dictionary;
            elseif check.same(dictionary_type,'Random:Filters')
                dictionary_ctor_fn_t = @transforms.record.dictionary.random.filters;
            elseif check.same(dictionary_type,'Random:Instances')
                dictionary_ctor_fn_t = @transforms.record.dictionary.random.instances;
            elseif check.same(dictionary_type,'Learn:Grad')
                dictionary_ctor_fn_t = @transforms.record.dictionary.learn.grad;
            elseif check.same(dictionary_type,'Learn:GradSt')
                dictionary_ctor_fn_t = @transforms.record.dictionary.learn.grad_st;
            else
                assert(false);
            end
            
            if check.same(nonlinear_type,'Linear')
                nonlinear_fn_t = @transforms.image.recoder.linear;
            elseif check.same(nonlinear_type,'Logistic')
                nonlinear_fn_t = @transforms.image.recoder.logistic;
            else
                assert(false);
            end

            if check.same(reduce_type,'Subsample')
                reduce_fn_t = @transforms.image.recoder.subsample;
                initial_value_t = 0;
            elseif check.same(reduce_type,'Sqr')
                reduce_fn_t = @transforms.image.recoder.sqr;
                initial_value_t = 0;
            elseif check.same(reduce_type,'Max')
                reduce_fn_t = @transforms.image.recoder.max;
                initial_value_t = 0;
            elseif check.same(reduce_type,'MaxMagnitude')
                reduce_fn_t = @transforms.image.recoder.max_magnitude;
                initial_value_t = 0;
            else
                assert(false);
            end
            
            t_patches = transforms.image.patch_extract(train_sample_plain,patches_count,patch_row_count,patch_col_count,patch_required_variance,logger.new_transform('Building patch extract transform'));
            patches_1 = t_patches.code(train_sample_plain,logger.new_transform('Extracting patches'));
            patches_2 = dataset.flatten_image(patches_1);
            patches_3 = bsxfun(@minus,patches_2,mean(patches_2,1));
            
            if do_patch_zca
                t_zca_t = transforms.record.zca(patches_3,logger.new_transform('Building ZCA transform'));
                patches_4 = t_zca_t.code(patches_3,logger.new_transform('Applying ZCA transform'));
            else
                t_zca_t = {};
                patches_4 = patches_3;
            end

            t_dictionary_first_t = dictionary_ctor_fn_t(patches_4,dictionary_params{:},do_polarity_split,logger.new_transform('Building patches dictionary'));
            
            if do_patch_zca
                back_projected_dict_t1 = t_zca_t.saved_transform_decode * t_dictionary_first_t.dict_transp;
                back_projected_dict = bsxfun(@plus,back_projected_dict_t1,t_zca_t.sample_mean)';
                % This might be problematic for funkier "dictionary_params".
                t_dictionary_t = transforms.record.dictionary(patches_4,back_projected_dict,dictionary_params{2},dictionary_params{3},do_polarity_split,logger.new_transform('Building ZCA backprojected dictionary'));
            else
                t_dictionary_t = t_dictionary_first_t;
            end
            
            input_geometry = [d,dr,dc,1];
            output_geometry = pooled_patch_row_count_t * pooled_patch_col_count_t * t_dictionary_t.output_geometry;
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.pooled_patch_row_count = pooled_patch_row_count_t;
            obj.pooled_patch_col_count = pooled_patch_col_count_t;
            obj.patch_count_row = patch_count_row_t;
            obj.patch_count_col = patch_count_col_t;
            obj.t_zca = t_zca_t;
            obj.t_dictionary = t_dictionary_t;
            obj.dictionary_ctor_fn = dictionary_ctor_fn_t;
            obj.nonlinear_fn = nonlinear_fn_t;
            obj.reduce_fn = reduce_fn_t;
            obj.initial_value = initial_value_t;
            obj.patches_count = patches_count;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.patch_required_variance = patch_required_variance;
            obj.do_patch_zca = do_patch_zca;
            obj.do_polarity_split = do_polarity_split;
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
            
            full_sample = zeros(obj.t_dictionary.output_geometry,N,obj.pooled_patch_row_count,obj.pooled_patch_col_count);
            local_sample_plain = shiftdim(sample_plain_padded,2);
            
            current_patch = 1;
            total_patches_count = obj.patch_count_row * obj.patch_count_col;
            
            for ii = 1:obj.pooled_patch_row_count
                for jj = 1:obj.pooled_patch_col_count
                    logger.beg_node('Output %02d%02d',ii,jj);

                    source_patches_row = ((ii - 1)*obj.reduce_spread + 1):(ii * obj.reduce_spread);
                    source_patches_col = ((jj - 1)*obj.reduce_spread + 1):(jj * obj.reduce_spread);
                    
                    local_sample = obj.initial_value * ones(obj.t_dictionary.output_geometry,N);
           
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
                            local_sample_4 = obj.t_dictionary.code(local_sample_3,logger);
                            local_sample_5 = obj.nonlinear_fn(local_sample_4,obj.nonlinear_params{:});

                            local_sample = obj.reduce_fn(local_sample,local_sample_5);

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
            if check.same(acc,zeros(size(acc)))
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
        
        function [o] = max_magnitude(acc,A)
            max_mask = abs(acc) > abs(A);
            o = max_mask .* acc + (~max_mask) .* A;
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.image.recoder".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    No ZCA, no polarity splitting, "Linear" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Subsample',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.subsample));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Subsample'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Linear" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Sqr',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.sqr));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Sqr'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Linear" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Max',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Max'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Linear" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'MaxMagnitude',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max_magnitude));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'MaxMagnitude'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Logistic" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Subsample',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.subsample));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Subsample'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Logistic" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Sqr',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.sqr));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Sqr'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Logistic" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Max',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Max'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Logistic" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'MaxMagnitude',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max_magnitude));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'MaxMagnitude'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Linear" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Subsample',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.subsample));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Subsample'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Linear" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Sqr',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.sqr));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Sqr'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Linear" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Max',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Max'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Linear" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'MaxMagnitude',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max_magnitude));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'MaxMagnitude'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Logistic" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Subsample',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.subsample));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Subsample'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Logistic" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Sqr',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.sqr));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Sqr'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Logistic" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Max',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Max'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Logistic" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'MaxMagnitude',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.same(t.t_zca,{}));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max_magnitude));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == false);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'MaxMagnitude'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Linear" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Sqr',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.sqr));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Sqr'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Linear" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Max',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Max'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Linear" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'MaxMagnitude',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max_magnitude));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'MaxMagnitude'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Logistic" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Subsample',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.subsample));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Subsample'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Logistic" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Sqr',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.sqr));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Sqr'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Logistic" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Max',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Max'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Logistic" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'MaxMagnitude',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == false);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,50));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max_magnitude));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == false);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'MaxMagnitude'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 50));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Linear" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Subsample',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.subsample));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Subsample'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Linear" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Sqr',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.sqr));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Sqr'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Linear" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Max',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Max'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Linear" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'MaxMagnitude',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.linear));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max_magnitude));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Linear'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'MaxMagnitude'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Logistic" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Subsample',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.subsample));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Subsample'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Logistic" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Sqr',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.sqr));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Sqr'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Logistic" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Max',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'Max'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Logistic" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            s = s(:,:,1,:);
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'MaxMagnitude',4,logg);
            
            assert(t.pooled_patch_row_count == 48);
            assert(t.pooled_patch_col_count == 64);
            assert(t.patch_count_row == 192);
            assert(t.patch_count_col == 256);
            assert(check.matrix(t.t_zca.saved_transform_code));
            assert(check.same(size(t.t_zca.saved_transform_code),[25 25]));
            assert(check.number(t.t_zca.saved_transform_code));
            assert(check.matrix(t.t_zca.saved_transform_decode));
            assert(check.same(size(t.t_zca.saved_transform_decode),[25 25]));
            assert(check.number(t.t_zca.saved_transform_decode));
            assert(check.matrix(t.t_zca.coeffs));
            assert(check.same(size(t.t_zca.coeffs),[25 25]));
            assert(check.number(t.t_zca.coeffs));
            assert(check.same(t.t_zca.coeffs * t.t_zca.coeffs',eye(25),0.1));
            assert(check.vector(t.t_zca.coeffs_eigenvalues));
            assert(length(t.t_zca.coeffs_eigenvalues) == 25);
            assert(check.number(t.t_zca.coeffs_eigenvalues));
            assert(check.vector(t.t_zca.sample_mean));
            assert(length(t.t_zca.sample_mean) == 25);
            assert(t.t_zca.div_epsilon == 0);
            assert(check.same(t.t_zca.input_geometry,25));
            assert(check.same(t.t_zca.output_geometry,25));
            assert(check.matrix(t.t_dictionary.dict));
            assert(check.same(size(t.t_dictionary.dict),[50 25]));
            assert(check.number(t.t_dictionary.dict));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict(ii,:)),1),1:50));
            assert(check.matrix(t.t_dictionary.dict_transp));
            assert(check.same(size(t.t_dictionary.dict_transp),[25 50]));
            assert(check.number(t.t_dictionary.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.t_dictionary.dict_transp(:,ii)),1),1:50));
            assert(check.same(t.t_dictionary.dict',t.t_dictionary.dict_transp));
            assert(t.t_dictionary.word_count == 50);
            assert(check.same(t.t_dictionary.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.t_dictionary.coding_params_cell,{}));
            assert(check.same(t.t_dictionary.coding_method,'Corr'));
            assert(check.same(t.t_dictionary.coding_params,[]));
            assert(t.t_dictionary.do_polarity_split == true);
            assert(check.same(t.t_dictionary.input_geometry,25));
            assert(check.same(t.t_dictionary.output_geometry,100));
            assert(check.same(t.dictionary_ctor_fn,@transforms.record.dictionary.random.instances));
            assert(check.same(t.nonlinear_fn,@transforms.image.recoder.logistic));
            assert(check.same(t.reduce_fn,@transforms.image.recoder.max_magnitude));
            assert(t.initial_value == 0);
            assert(t.patches_count == 1000);
            assert(t.patch_row_count == 5);
            assert(t.patch_col_count == 5);
            assert(t.patch_required_variance == 0.01);
            assert(t.do_patch_zca == true);
            assert(t.do_polarity_split == true);
            assert(check.same(t.dictionary_type,'Random:Instances'));
            assert(check.same(t.dictionary_params,{50 'Corr' {}}));
            assert(t.window_step == 1);
            assert(check.same(t.nonlinear_type,'Logistic'));
            assert(check.same(t.nonlinear_params,{}));
            assert(check.same(t.reduce_type,'MaxMagnitude'));
            assert(t.reduce_spread == 4);
            assert(check.same(t.input_geometry,[192 * 256 192 256 1]));
            assert(check.same(t.output_geometry,48 * 64 * 100));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            fprintf('    No ZCA, no polarity splitting, "Linear" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Subsample',4,logg);
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Linear" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Sqr',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Linear" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Max',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Linear" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'MaxMagnitude',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Logistic" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Subsample',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Logistic" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Sqr',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Logistic" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Max',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, no polarity splitting, "Logistic" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'MaxMagnitude',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Linear" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Subsample',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Linear" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Subsample',4,logg);
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Linear" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Sqr',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Linear" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Max',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Linear" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'MaxMagnitude',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Logistic" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Subsample',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Logistic" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Sqr',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Logistic" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Max',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    No ZCA, with polarity splitting, "Logistic" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,false,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'MaxMagnitude',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Linear" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Subsample',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Linear" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Sqr',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Linear" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Max',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Linear" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'MaxMagnitude',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Logistic" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Subsample',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Logistic" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Sqr',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Logistic" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Max',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, no polarity splitting, "Logistic" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,false,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'MaxMagnitude',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 50 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,false,7,7,1);
                pause(5);
            end
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Linear" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Subsample',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Linear" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Sqr',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Linear" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'Max',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Linear" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Linear',{},'MaxMagnitude',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Logistic" nonlinearity and "Subsample" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Subsample',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Logistic" nonlinearity and "Sqr" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Sqr',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Logistic" nonlinearity and "Max" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'Max',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With ZCA, with polarity splitting, "Logistic" nonlinearity and "MaxMagnitude" reduce type.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.mnist();
            
            t = transforms.image.recoder(s,1000,5,5,0.01,true,true,'Random:Instances',{50 'Corr' {}},1,'Logistic',{},'MaxMagnitude',4,logg);
            
            s_p = t.code(s(:,:,:,1:5),logg);
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[7 * 7 * 100 5]));
            assert(check.number(s_p));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.coded_output(s(:,:,:,1:5),s_p,50,true,7,7,1);
                pause(5);
            end
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
        end
    end
end
