classdef grad_st < transforms.record.dictionary
    properties (GetAccess=public,SetAccess=immutable)
        saved_mse;
        selection_size;
        initial_learning_rate;
        final_learning_rate;
        max_iter_count;
    end

    methods (Access=public)
        function [obj] = grad_st(train_sample_plain,word_count,coding_method,coding_params,selection_size,initial_learning_rate,final_learning_rate,max_iter_count,logger)
            assert(tc.dataset_record(train_sample_plain));
            assert(tc.scalar(word_count));
            assert(tc.natural(word_count));
            assert(word_count > 0);
            assert(tc.scalar(coding_method));
            assert(tc.string(coding_method));
            assert(tc.one_of(coding_method,'Corr','MP','OMP','SparseNet','Euclidean'));
            assert((tc.same(coding_method,'Corr') && tc.empty(coding_params)) || ...
                   (tc.same(coding_method,'MP') && (tc.scalar(coding_params) && ...
                                                    tc.natural(coding_params) && ...
                                                    (coding_params > 0) && ...
                                                    (coding_params <= word_count))) || ...
                   (tc.same(coding_method,'SparseNet') && (tc.scalar(coding_params) && ...
                                                           tc.unitreal(coding_params) && ...
                                                           (coding_params > 0))) || ...                                 
                   (tc.same(coding_method,'OMP') && (tc.scalar(coding_params) && ...
                                                     tc.natural(coding_params) && ...
                                                     (coding_params > 0) && ...
                                                     (coding_params <= word_count))) || ...
                   (tc.same(coding_method,'Euclidean') && tc.empty(coding_params)));
            assert(tc.scalar(selection_size));
            assert(tc.natural(selection_size));
            assert(selection_size >= 1);
            assert(selection_size <= dataset.count(train_sample_plain));
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
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            if tc.same(coding_method,'Corr')
                coding_fn_t = @transforms.record.dictionary.correlation;
                coding_params_cell_t = {};
            elseif tc.same(coding_method,'MP')
                coding_fn_t = @transforms.record.dictionary.matching_pursuit;
                coding_params_cell_t = {coding_params(1)};
            elseif tc.same(coding_method,'OMP')
                coding_fn_t = @transforms.record.dictionary.ortho_matching_pursuit;
                coding_params_cell_t = {coding_params(1)};
            elseif tc.same(coding_method,'SparseNet')
                coding_fn_t = @transforms.record.dictionary.sparse_net;
                coding_params_cell_t = {coding_params(1)};
            else
                coding_fn_t = @transforms.record.dictionary.euclidean;
                coding_params_cell_t = {};
            end
            
            N = dataset.count(train_sample_plain);
            d = dataset.geometry(train_sample_plain);
            dict = transforms.record.dictionary.normalize_dict(utils.rand_range(-1,1,word_count,d));
            dict_transp = dict';
            saved_mse_t = zeros(1,max_iter_count);
            
            logger.beg_node('Learning sparse dictionary');

            for iter = 1:max_iter_count
                logger.message('Iteration %d.',iter);
                
                selected = randi(N,1,selection_size);
                iter_sample = dataset.subsample(train_sample_plain,selected);
                
                coeffs = coding_fn_t(dict,dict_transp,iter_sample,coding_params_cell_t{:});
                
                diff = iter_sample - dict_transp * coeffs;
                delta_dict = coeffs * diff';
                learning_rate = initial_learning_rate * (final_learning_rate / initial_learning_rate) ^ (iter / max_iter_count);
                
                dict = dict + learning_rate / selection_size * delta_dict;
                dict = transforms.record.dictionary.normalize_dict(dict);
                dict_transp = dict';
                
                mean_error = sum(mean((diff .^ 2)));
                saved_mse_t(iter) = mean_error;
                logger.message('Mean error: %.0f',mean_error);
                if mod(iter - 1,1) == 0
                    sz = sqrt(size(dict,2));
                    subplot(2,1,1);
                    utilsdisplay.sparse_basis(dict,sz,sz);
                    subplot(2,1,2);
                    plot(saved_mse_t);
                    axis([1 max_iter_count 0 max(saved_mse_t)]);
                    pause(0.1);
                end
            end
            
            logger.end_node();
            
            obj = obj@transforms.record.dictionary(train_sample_plain,dict,coding_method,coding_params,logger);
            obj.saved_mse = saved_mse_t;
            obj.selection_size = selection_size;
            obj.initial_learning_rate = initial_learning_rate;
            obj.final_learning_rate = final_learning_rate;
            obj.max_iter_count = max_iter_count;
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "transforms.record.dictionary.learn.grad_st".\n');
            
            fprintf('  Proper constuction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary.learn.grad_st(s,3,'Corr',[],10,1e-2,1e-4,100,log);
            
            assert(tc.vector(t.saved_mse));
            assert(length(t.saved_mse) == 100);
            assert(tc.number(t.saved_mse));
            assert(tc.check(t.saved_mse > 0));
            assert(t.selection_size == 10);
            assert(t.initial_learning_rate == 1e-2);
            assert(t.final_learning_rate == 1e-4);
            assert(t.max_iter_count == 100);
            assert(tc.matrix(t.dict));
            assert(tc.same(size(t.dict),[3 2]));
            assert(tc.number(t.dict));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict(ii,:)),1),1:3));
            assert(tc.matrix(t.dict_transp));
            assert(tc.same(size(t.dict_transp),[2 3]));
            assert(tc.number(t.dict_transp));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict_transp(:,ii)),1),1:3));
            assert(tc.same(t.dict_transp,t.dict'));
            assert(t.word_count == 3);
            assert(tc.same(t.coding_fn,@transforms.record.dictionary.correlation));
            assert(tc.same(t.coding_params_cell,{}));
            assert(tc.same(t.coding_method,'Corr'));
            assert(tc.same(t.coding_params,[]));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,3));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
