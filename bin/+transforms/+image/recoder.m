classdef recoder < transform
    properties (GetAccess=public,SetAccess=immutable)
        dictionary_ctor_fn;
        word_count;
        coding_code;
        aftcoding_row_count;
        aftcoding_col_count;
        nonlinear_code;
        nonlinear_modulator;
        polarity_split_code;
        polarity_split_multiplier;
        reduce_code;
        aftreduce_row_count;
        aftreduce_col_count;
        t_zca;
        t_dictionary;
        patches_count;
        patch_row_count;
        patch_col_count;
        patch_required_variance;
        do_patch_zca;
        dictionary_type;
        dictionary_params;
        nonlinear_type;
        nonlinear_params;
        polarity_split_type;
        reduce_type;
        reduce_spread;
        num_workers;
    end
    
    methods (Access=public)
        function [obj] = recoder(train_sample_plain,patches_count,patch_row_count,patch_col_count,patch_required_variance,do_patch_zca,...
                                 dictionary_type,dictionary_params,nonlinear_type,nonlinear_params,polarity_split_type,reduce_type,reduce_spread,num_workers)
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
            assert(check.scalar(dictionary_type));
            assert(check.string(dictionary_type));
            assert(check.one_of(dictionary_type,'Dict','Random:Filters','Random:Instances','Learn:Grad','Learn:GradSt'));
            assert((check.same(dictionary_type,'Dict') && check.matrix(dictionary_params{1}) && ...
                   (size(dictionary_params{1},2) == patch_row_count * patch_col_count) && check.number(dictionary_params{1})) || ...
                   (check.one_of(dictionary_type,'Random:Filters','Random:Instances','Learn:Grad','Learn:GradSt') && ...
                    check.scalar(dictionary_params{1}) && check.natural(dictionary_params{1}) && dictionary_params{1} >= 1));
            assert(check.vector(dictionary_params));
            assert(length(dictionary_params) >= 4);
            assert(check.cell(dictionary_params));
            assert(check.scalar(dictionary_params{2}));
            assert(check.string(dictionary_params{2}));
            assert(check.one_of(dictionary_params{2},'Corr','MP'));
            assert(check.scalar(dictionary_params{4}));
            assert(check.natural(dictionary_params{4}));
            assert(dictionary_params{4} >= 1);
            assert(check.scalar(dictionary_params{5}));
            assert(check.natural(dictionary_params{5}));
            assert(dictionary_params{5} >= 1);
            assert(check.scalar(nonlinear_type));
            assert(check.string(nonlinear_type));
            assert(check.one_of(nonlinear_type,'Linear','Logistic','GlobalOrder'));
            assert((check.same(nonlinear_type,'Linear') && check.empty(nonlinear_params)) || ...
                   (check.same(nonlinear_type,'Logistic') && check.empty(nonlinear_params)) || ...
                   (check.same(nonlinear_type,'GlobalOrder') && check.scalar(nonlinear_params) && ...
                    check.number(nonlinear_params) && (nonlinear_params > 0)));
            assert(check.scalar(polarity_split_type));
            assert(check.string(polarity_split_type));
            assert(check.one_of(polarity_split_type,'None','NoSign','KeepSign'));
            assert(check.scalar(reduce_type));
            assert(check.string(reduce_type));
            assert(check.one_of(reduce_type,'Subsample','MaxNoSign','MaxKeepSign','SumAbs','SumSqr'));
            assert(check.scalar(reduce_spread));
            assert(check.natural(reduce_spread));
            assert(reduce_spread >= 1);
            assert(check.scalar(num_workers));
            assert(check.natural(num_workers));
            assert(num_workers >= 1);
            
            [d,dr,dc,~] = dataset.geometry(train_sample_plain);
            
            if check.same(dictionary_type,'Dict')
                dictionary_ctor_fn_t = @transforms.record.dictionary;
                word_count_t = size(dictionary_params{1},1);
            elseif check.same(dictionary_type,'Random:Filters')
                dictionary_ctor_fn_t = @transforms.record.dictionary.random.filters;
                word_count_t = dictionary_params{1};
            elseif check.same(dictionary_type,'Random:Instances')
                dictionary_ctor_fn_t = @transforms.record.dictionary.random.instances;
                word_count_t = dictionary_params{1};
            elseif check.same(dictionary_type,'Learn:Grad')
                dictionary_ctor_fn_t = @transforms.record.dictionary.learn.grad;
                word_count_t = dictionary_params{1};
            elseif check.same(dictionary_type,'Learn:GradSt')
                dictionary_ctor_fn_t = @transforms.record.dictionary.learn.grad_st;
                word_count_t = dictionary_params{1};
            else
                assert(false);
            end
            
            if check.same(dictionary_params{2},'Corr')
                coding_code_t = 0;
            elseif check.same(dictionary_params{2},'MP')
                coding_code_t = 1;
            else
                assert(false);
            end
            
            aftcoding_row_count_t = dr - mod(dr,reduce_spread);
            aftcoding_col_count_t = dc - mod(dc,reduce_spread);
            
            if check.same(nonlinear_type,'Linear')
                nonlinear_code_t = 0;
                nonlinear_modulator_t = [];
            elseif check.same(nonlinear_type,'Logistic')
                nonlinear_code_t = 1;
                nonlinear_modulator_t = [];
            elseif check.same(nonlinear_type,'GlobalOrder')
                nonlinear_code_t = 2;
                nonlinear_modulator_t = utils.common.schedule(1,nonlinear_params,dictionary_params{4});
            else
                assert(false);
            end
            
            if check.same(polarity_split_type,'None')
                polarity_split_code_t = 0;
                polarity_split_multiplier_t = 1;
            elseif check.same(polarity_split_type,'NoSign')
                polarity_split_code_t = 1;
                polarity_split_multiplier_t = 2;
            elseif check.same(polarity_split_type,'KeepSign')
                polarity_split_code_t = 2;
                polarity_split_multiplier_t = 2;
            else
                assert(false);
            end
            
            if check.same(reduce_type,'Subsample')
                reduce_code_t = 0;
            elseif check.same(reduce_type,'MaxNoSign')
                reduce_code_t = 1;
            elseif check.same(reduce_type,'MaxKeepSign')
                reduce_code_t = 2;
            elseif check.same(reduce_type,'SumAbs')
                reduce_code_t = 3;
            elseif check.same(reduce_type,'SumSqr')
                reduce_code_t = 4;
            else
                assert(false);
            end
            
            aftreduce_row_count_t = aftcoding_row_count_t / reduce_spread;
            aftreduce_col_count_t = aftcoding_col_count_t / reduce_spread;
            
            t_patches = transforms.image.patch_extract(train_sample_plain,patches_count,patch_row_count,patch_col_count,patch_required_variance);
            patches_1 = t_patches.code(train_sample_plain);
            patches_2 = dataset.flatten_image(patches_1);
            patches_3 = bsxfun(@minus,patches_2,mean(patches_2,1));
            
            if do_patch_zca
                t_zca_t = transforms.record.zca(patches_3);
                patches_4 = t_zca_t.code(patches_3);
            else
                t_zca_t = {};
                patches_4 = patches_3;
            end

            t_dictionary_t = dictionary_ctor_fn_t(patches_4,dictionary_params{:});
            
            input_geometry = [d,dr,dc,1];
            output_geometry = polarity_split_multiplier_t * aftreduce_row_count_t * aftreduce_col_count_t * word_count_t;
            
            obj = obj@transform(input_geometry,output_geometry);
            obj.dictionary_ctor_fn = dictionary_ctor_fn_t;
            obj.word_count = word_count_t;
            obj.coding_code = coding_code_t;
            obj.aftcoding_row_count = aftcoding_row_count_t;
            obj.aftcoding_col_count = aftcoding_col_count_t;
            obj.nonlinear_code = nonlinear_code_t;
            obj.nonlinear_modulator = nonlinear_modulator_t;
            obj.polarity_split_code = polarity_split_code_t;
            obj.polarity_split_multiplier = polarity_split_multiplier_t;
            obj.reduce_code = reduce_code_t;
            obj.aftreduce_row_count = aftreduce_row_count_t;
            obj.aftreduce_col_count = aftreduce_col_count_t;
            obj.t_zca = t_zca_t;
            obj.t_dictionary = t_dictionary_t;
            obj.patches_count = patches_count;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.patch_required_variance = patch_required_variance;
            obj.do_patch_zca = do_patch_zca;
            obj.dictionary_type = dictionary_type;
            obj.dictionary_params = dictionary_params;
            obj.nonlinear_type = nonlinear_type;
            obj.nonlinear_params = nonlinear_params;
            obj.polarity_split_type = polarity_split_type;
            obj.reduce_type = reduce_type;
            obj.reduce_spread = reduce_spread;
            obj.num_workers = num_workers;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain)
            [d,dr,dc,~] = dataset.geometry(sample_plain);
            sample_plain_flattened = reshape(sample_plain,d,[]);
            [sample_coded_t,observations_perm] = ...
                xtern.x_image_recoder_code(dr,dc,obj.patch_row_count,obj.patch_col_count,...
                                           obj.coding_code,obj.t_dictionary.dict,obj.t_dictionary.dict_transp,obj.t_dictionary.dict_x_dict_transp,...
                                           obj.t_dictionary.coeff_count,obj.t_dictionary.coding_params_cell{:},...
                                           obj.nonlinear_code,obj.nonlinear_modulator,obj.polarity_split_code,obj.reduce_code,obj.reduce_spread,...
                                           sample_plain_flattened,obj.num_workers);
            sample_coded(:,observations_perm+1) = sample_coded_t;
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.image.recoder".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    NOT YET TESTED.\n');
            
            fprintf('  Function "code".\n');
            
            fprintf('    NOT YET TESTED.\n');
        end
    end
end
