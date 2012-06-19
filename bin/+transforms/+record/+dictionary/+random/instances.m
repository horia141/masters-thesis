classdef instances < transforms.record.dictionary
    methods (Access=public)
        function [obj] = instances(train_sample_plain,word_count,coding_method,coding_params,logger)
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
                   (tc.same(coding_method,'OMP') && (tc.scalar(coding_params) && ...
                                                     tc.natural(coding_params) && ...
                                                     (coding_params > 0) && ...
                                                     (coding_params <= word_count))) || ...
                   (tc.same(coding_method,'SparseNet') && (tc.scalar(coding_params) && ...
                                                           tc.unitreal(coding_params) && ...
                                                           (coding_params > 0))) || ...
                   (tc.same(coding_method,'Euclidean') && tc.empty(coding_params)));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(dataset.count(train_sample_plain) >= word_count);
            
            N = dataset.count(train_sample_plain);
            dict = dataset.subsample(train_sample_plain,randi(N,1,word_count))';
            
            obj = obj@transforms.record.dictionary(train_sample_plain,dict,coding_method,coding_params,logger);
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "transforms.record.dictionary.random.instances".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary.random.instances(s,3,'Corr',[],log);
            
            assert(tc.matrix(t.dict));
            assert(tc.same(size(t.dict),[3 2]));
            assert(tc.number(t.dict));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict(ii,:)),1),1:3));
            assert(tc.matrix(t.dict_transp));
            assert(tc.same(size(t.dict_transp),[2 3]));
            assert(tc.number(t.dict_transp));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict_transp(:,ii)),1),1:3));
            assert(tc.same(t.dict',t.dict_transp));
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
