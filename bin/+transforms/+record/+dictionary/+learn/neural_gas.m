classdef neural_gas < transforms.record.dictionary
    properties (GetAccess=public,SetAccess=immutable)
        saved_mse;
        version;
        initial_learn_rate;
        final_learn_rate;
        initial_neight_size;
        final_neight_size;
        max_iter_count;
    end
    
    methods (Access=public)
        function [obj] = neural_gas(train_sample_plain,word_count,coding_method,coding_params,coeff_count,num_workers,version,initial_learn_rate,final_learn_rate,initial_neight_size,final_neight_size,max_iter_count)
            assert(check.dataset_record(train_sample_plain));
            assert(check.scalar(word_count));
            assert(check.natural(word_count));
            assert(word_count >= 1);
            assert(transforms.record.dictionary.coding_setup_ok(coding_method,coding_params));
            assert(check.scalar(coeff_count));
            assert(check.natural(coeff_count));
            assert(coeff_count >= 1);
            assert(coeff_count <= word_count);
            assert(check.scalar(num_workers));
            assert(check.natural(num_workers));
            assert(num_workers >= 1);
            assert(check.scalar(version));
            assert(check.string(version));
            assert(check.one_of(version,'V1','V2'));
            assert(check.scalar(initial_learn_rate));
            assert(check.number(initial_learn_rate));
            assert(initial_learn_rate > 0);
            assert(check.scalar(final_learn_rate));
            assert(check.number(final_learn_rate));
            assert(final_learn_rate > 0);
            assert(check.scalar(initial_neight_size));
            assert(check.number(initial_neight_size));
            assert(initial_neight_size > 0);
            assert(check.scalar(final_neight_size));
            assert(check.number(final_neight_size));
            assert(final_neight_size > 0);
            assert(final_learn_rate <= initial_learn_rate);
            assert(final_neight_size <= initial_neight_size);

            coding_fn_t = transforms.record.dictionary.coding_setup(coding_method);

            N = dataset.count(train_sample_plain);
            d = dataset.geometry(train_sample_plain);
            dict = transforms.record.dictionary.normalize_dict(utils.common.rand_range(-1,1,word_count,d));
            dict_transp = dict';
            dict_x_dict_transp = dict * dict_transp;
            saved_mse_t = zeros(1,max_iter_count);
            learn_rate_sch = utils.common.schedule(initial_learn_rate,final_learn_rate,max_iter_count);
            neight_size_sch = utils.common.schedule(initial_neight_size,final_neight_size,max_iter_count);
            ranks = zeros(1,word_count);

            if check.same(version,'V1')
                for iter = 1:max_iter_count
                    target_observation = train_sample_plain(:,randi(N));
                    
                    similarities = coding_fn_t(dict,dict_transp,dict_x_dict_transp,coding_params,coeff_count,target_observation,num_workers);
                    [~,sorted_similarities_idx] = sort(-abs(similarities),'ascend');
                    ranks(sorted_similarities_idx) = 1:word_count;
                    similarities_big = repmat(similarities,1,d);
                    diff = repmat(target_observation',word_count,1) - similarities_big .* dict;
                    scaling = repmat((2.71828183 .^ (-ranks / neight_size_sch(iter)))',1,d);
                    dict = dict + learn_rate_sch(iter) * scaling .* similarities_big .* diff;
                    dict = transforms.record.dictionary.normalize_dict(dict);
                    dict_transp = dict';
                    
                    saved_mse_t(iter) = norm(target_observation - dict_transp * similarities);
                    
%                     if mod(iter-1,1000) == 0
%                         clf(gcf());
%                         subplot(2,1,1);
%                         utils.display.dictionary(dict,9,9);
%                         subplot(2,1,2);
%                         plot(saved_mse_t);
%                         pause(0.1);
%                     end
                end
            elseif check.same(version,'V2');
                assert(false);
            else
                assert(false);
            end
            
            obj = obj@transforms.record.dictionary(train_sample_plain,dict,coding_method,coding_params,coeff_count,num_workers);
            obj.saved_mse = saved_mse_t;
            obj.version = version;
            obj.initial_learn_rate = initial_learn_rate;
            obj.final_learn_rate = final_learn_rate;
            obj.initial_neight_size = initial_neight_size;
            obj.final_neight_size = final_neight_size;
            obj.max_iter_count = max_iter_count;
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.dictionary.learn.neural_gas".\n');
            
            fprintf('  Proper construction.\n');
        end
    end
end
