classdef grad_st < transforms.record.dictionary
    properties (GetAccess=public,SetAccess=immutable)
        saved_mse;
        selection_size;
        initial_learning_rate;
        final_learning_rate;
        max_iter_count;
    end

    methods (Access=public)
        function [obj] = grad_st(train_sample_plain,word_count,coding_method,coding_params,coeff_count,num_workers,selection_size,initial_learning_rate,final_learning_rate,max_iter_count)
            assert(check.dataset_record(train_sample_plain));
            assert(check.scalar(word_count));
            assert(check.natural(word_count));
            assert(word_count > 0);
            assert(transforms.record.dictionary.coding_setup_ok(coding_method,coding_params));
            assert(check.scalar(coeff_count));
            assert(check.natural(coeff_count));
            assert(coeff_count >= 1);
            assert(coeff_count <= word_count);
            assert(check.scalar(num_workers));
            assert(check.natural(num_workers));
            assert(num_workers >= 1);
            assert(check.scalar(selection_size));
            assert(check.natural(selection_size));
            assert(selection_size >= 1);
            assert(selection_size <= dataset.count(train_sample_plain));
            assert(check.scalar(initial_learning_rate));
            assert(check.number(initial_learning_rate));
            assert(initial_learning_rate > 0);
            assert(check.scalar(final_learning_rate));
            assert(check.number(final_learning_rate));
            assert(final_learning_rate > 0);
            assert(final_learning_rate <= initial_learning_rate);
            assert(check.scalar(max_iter_count));
            assert(check.natural(max_iter_count));
            assert(max_iter_count >= 1);

            coding_fn_t = transforms.record.dictionary.coding_setup(coding_method,coding_params);
            
            N = dataset.count(train_sample_plain);
            d = dataset.geometry(train_sample_plain);
            dict = transforms.record.dictionary.normalize_dict(utils.common.rand_range(-1,1,word_count,d));
            dict_transp = dict';
            dict_x_dict_transp = dict * dict_transp;
            saved_mse_t = zeros(1,max_iter_count);
            learning_rate_schedule = utils.common.schedule(initial_learning_rate,final_learning_rate,max_iter_count);
            
            for iter = 1:max_iter_count
                selected = randi(N,1,selection_size);
                iter_sample = dataset.subsample(train_sample_plain,selected);

                coeffs = coding_fn_t(dict,dict_transp,dict_x_dict_transp,coding_params,coeff_count,iter_sample,num_workers);

                diff = iter_sample - dict_transp * coeffs;
                delta_dict = coeffs * diff';

                dict = dict + (learning_rate_schedule(iter) / selection_size) * delta_dict;
                dict = transforms.record.dictionary.normalize_dict(dict);
                dict_transp = dict';
                
                mean_error = sum(mean((diff .^ 2)));
                saved_mse_t(iter) = mean_error;
                if mod(iter - 1,100) == 0
                    sz = sqrt(size(dict,2));
                    subplot(2,1,1);
                    utils.display.dictionary(dict,sz,sz);
                    subplot(2,1,2);
                    plot(saved_mse_t);
                    axis([1 max_iter_count 0 max(saved_mse_t)]);
                    pause(0.1);
                end
            end
            
            obj = obj@transforms.record.dictionary(train_sample_plain,dict,coding_method,coding_params,coeff_count,num_workers);
            obj.saved_mse = saved_mse_t;
            obj.selection_size = selection_size;
            obj.initial_learning_rate = initial_learning_rate;
            obj.final_learning_rate = final_learning_rate;
            obj.max_iter_count = max_iter_count;
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.dictionary.learn.grad_st".\n');
            
            fprintf('  Proper constuction.\n');
            
            s = dataset.load('../../test/three_component_cloud.mat');

            t = transforms.record.dictionary.learn.grad_st(s,3,'MP',[],1,2,1,1,1e-2,100);
            
            assert(check.vector(t.saved_mse));
            assert(length(t.saved_mse) == 100);
            assert(check.number(t.saved_mse));
            assert(check.checkv(t.saved_mse > 0));
            assert(t.selection_size == 1);
            assert(t.initial_learning_rate == 1);
            assert(t.final_learning_rate == 1e-2);
            assert(t.max_iter_count == 100);
            assert(check.matrix(t.dict));
            assert(check.same(size(t.dict),[3 2]));
            assert(check.number(t.dict));
            assert(check.checkf(@(ii)check.same(norm(t.dict(ii,:)),1),1:3));
            assert(check.matrix(t.dict_transp));
            assert(check.same(size(t.dict_transp),[2 3]));
            assert(check.number(t.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.dict_transp(:,ii)),1),1:3));
            assert(check.same(t.dict_transp,t.dict'));
            assert(t.word_count == 3);
            assert(check.same(t.coding_fn,@xtern.x_dictionary_matching_pursuit));
            assert(check.same(t.coding_params_cell,{[]}));
            assert(check.same(t.coding_method,'MP'));
            assert(check.same(t.coding_params,[]));
            assert(t.coeff_count == 1);
            assert(t.num_workers == 2);
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,3));

            s_p = t.code(s);
            s_r = t.dict_transp * s_p;

            if test_figure ~= -1
                figure(test_figure);
                clf(gcf());
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                hold off;
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
            end
            
            clearvars -except test_figure;
        end
    end
end
