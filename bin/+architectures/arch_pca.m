classdef arch_pca < architecture
    properties (GetAccess=public,SetAccess=immutable)
        kept_energy;
    end
    
    methods (Access=public)
        function [obj] = arch_pca(train_dataset_plain,kept_energy,classifier_ctor_fn,classifier_params,logger)
            assert(tc.scalar(train_dataset_plain));
            assert(tc.dataset(train_dataset_plain));
            assert(train_dataset_plain.samples_count >= 1);
            assert(tc.scalar(kept_energy));
            assert(tc.unitreal(kept_energy));
            assert(tc.scalar(classifier_ctor_fn));
            assert(tc.function_h(classifier_ctor_fn));
            assert(tc.empty(classifier_params) || tc.vector(classifier_params));
            assert(tc.empty(classifier_params) || tc.cell(classifier_params));
            assert(tc.empty(classifier_params) || tc.checkf(@tc.value,classifier_params));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            t_pca = transforms.pca(train_dataset_plain,kept_energy,logger.new_transform('Training PCA transform'));
            train_dataset_1 = t_pca.code(train_dataset_plain,logger.new_transform('Transforming training dataset'));
            
            classifier = classifier_ctor_fn(train_dataset_1,classifier_params{:},logger.new_classifier('Training classifier'));
            
            obj = obj@architecture(train_dataset_plain.subsamples(1),{t_pca},classifier,logger);
            obj.kept_energy = kept_energy;
        end
    end
end
