classdef neural_gas < transforms.record.dictionary
    properties (GetAccess=public,SetAccess=immutable)
        actual_iter_count;
        conn_graph;
        eta_1;
        eta_f;
        lambda_1;
        lambda_f;
        timeout_1;
        timeout_f;
        max_iter_count;
    end

    methods (Access=public)
        function [obj] = neural_gas(train_sample_plain,word_count,coding_method,coding_params,eta_1,eta_f,lambda_1,lambda_f,timeout_1,timeout_f,max_iter_count,logger)
            assert(tc.dataset_record(train_sample_plain));
            assert(tc.scalar(word_count));
            assert(tc.natural(word_count));
            assert(word_count >= 2);
            assert(tc.scalar(coding_method));
            assert(tc.string(coding_method));
            assert(tc.one_of(coding_method,'Corr','MP','OMP','SparseNet','Euclidean'));
            assert((tc.same(coding_method,'Corr') && tc.empty(coding_params)) || ...
                   (tc.same(coding_method,'MP') && (tc.scalar(coding_params) && ...
                                                    tc.natural(coding_params) && ...
                                                    (coding_params > 0) && ...
                                                    (coding_params <= word_count))) || ...
                   (tc.same(coding_method,'OMP') && (tc.scalar(coding_params) && ...
                                                     tc.natural(coding_params) && ...
                                                     (coding_params > 0) && ...
                                                     (coding_params <= word_count))) || ...
                   (tc.same(coding_method,'SparseNet') && (tc.scalar(coding_params) && ...
                                                           tc.unitreal(coding_params) && ...
                                                           (coding_params > 0))) || ...
                   (tc.same(coding_method,'Euclidean') && tc.empty(coding_params)));
            assert(tc.scalar(eta_1));
            assert(tc.number(eta_1));
            assert(eta_1 > 0);
            assert(tc.scalar(eta_f));
            assert(tc.number(eta_f));
            assert(eta_f > 0);
            assert(tc.scalar(lambda_1));
            assert(tc.number(lambda_1));
            assert(lambda_1 > 0);
            assert(tc.scalar(lambda_f));
            assert(tc.number(lambda_f));
            assert(lambda_f > 0);
            assert(tc.scalar(timeout_1));
            assert(tc.natural(timeout_1));
            assert(timeout_1 > 0);
            assert(tc.scalar(timeout_f));
            assert(tc.natural(timeout_f));
            assert(timeout_f > 0);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(eta_1 > eta_f);
            assert(lambda_1 > lambda_f);
            
            N = dataset.count(train_sample_plain);
            d = dataset.geometry(train_sample_plain);
            dict_transp = train_sample_plain(:,randi(N,1,word_count));
            conn_graph_t = false(word_count,word_count);
            conn_age = zeros(word_count,word_count);
            
            actual_iter_count_f = max_iter_count * N;
            
            eta = eta_1 * (eta_f / eta_1) .^ ((1:actual_iter_count_f) / actual_iter_count_f);
            lambda = lambda_1 * (lambda_f / lambda_1) .^ ((1:actual_iter_count_f) / actual_iter_count_f);
            timeout = timeout_1 * (timeout_f / timeout_1) .^ ((1:actual_iter_count_f) / actual_iter_count_f);
            
            for iter = 1:actual_iter_count_f
                if mod(iter - 1,N) == 0
                    logger.message('Iteration %d.',(iter - 1) / N + 1);
                end

                instance = train_sample_plain(:,randi(N));
                diff_dict = bsxfun(@minus,dict_transp,instance);
                distances = sum(diff_dict .^ 2,1);
                [~,distances_sorted_orig] = sort(distances);
                orders = repmat(2.71828183 .^ arrayfun(@(a)-sum(a > distances)/lambda(iter),distances),d,1);

                dict_transp = dict_transp - eta(iter) * orders .* diff_dict;

                conn_graph_t(conn_age > timeout(iter)) = false;
                conn_age = conn_age + 1;
                conn_graph_t(distances_sorted_orig(1),distances_sorted_orig(2)) = true;
                conn_graph_t(distances_sorted_orig(2),distances_sorted_orig(1)) = true;
                conn_age(distances_sorted_orig(1),distances_sorted_orig(2)) = 0;
                conn_age(distances_sorted_orig(2),distances_sorted_orig(1)) = 0;
            end
            
            obj = obj@transforms.record.dictionary(train_sample_plain,dict_transp',coding_method,coding_params,logger);
            obj.actual_iter_count = actual_iter_count_f;
            obj.conn_graph = conn_graph_t;
            obj.eta_1 = eta_1;
            obj.eta_f = eta_f;
            obj.lambda_1 = lambda_1;
            obj.lambda_f = lambda_f;
            obj.timeout_1 = timeout_1;
            obj.timeout_f = timeout_f;
            obj.max_iter_count = max_iter_count;
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "transforms.record.dictionary.learn.neural_gas".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary.learn.neural_gas(s,10,'Corr',[],1e-2,1e-5,20,0.5,20,200,10,log);
            
            assert(t.actual_iter_count == 6000);
            assert(tc.matrix(t.conn_graph));
            assert(tc.logical(t.conn_graph));
            assert(tc.same(arrayfun(@(ii)t.conn_graph(ii,ii),1:10),false(1,10)));
            assert(t.eta_1 == 1e-2);
            assert(t.eta_f == 1e-5);
            assert(t.lambda_1 == 20);
            assert(t.lambda_f == 0.5);
            assert(t.timeout_1 == 20);
            assert(t.timeout_f == 200);
            assert(t.max_iter_count == 10);
            assert(tc.matrix(t.dict));
            assert(tc.same(size(t.dict),[10 2]));
            assert(tc.number(t.dict));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict(ii,:)),1),1:10));
            assert(tc.matrix(t.dict_transp));
            assert(tc.same(size(t.dict_transp),[2 10]));
            assert(tc.number(t.dict_transp));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict_transp(:,ii)),1),1:10));
            assert(tc.same(t.dict',t.dict_transp));
            assert(t.word_count == 10);
            assert(tc.same(t.coding_fn,@transforms.record.dictionary.correlation));
            assert(tc.same(t.coding_params_cell,{}));
            assert(tc.same(t.coding_method,'Corr'));
            assert(tc.same(t.coding_params,[]));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,10));

            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end