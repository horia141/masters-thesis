classdef arch_pca < architecture
    properties (GetAccess=public,SetAccess=immutable)
        kept_energy;
    end
    
    methods (Access=public)
        function [obj] = arch_pca(train_sample_plain,class_info,kept_energy,classifier_ctor_fn,classifier_params,logger)
            assert(tc.dataset_record(train_sample_plain));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
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
            assert(class_info.compatible(train_sample_plain));
            
            t_pca = transforms.pca(train_sample_plain,kept_energy,logger.new_transform('Training PCA transform'));
            train_dataset_1 = t_pca.code(train_sample_plain,logger.new_transform('Transforming training dataset'));
            
            classifier = classifier_ctor_fn(train_dataset_1,class_info,classifier_params{:},logger.new_classifier('Training classifier'));
            
            obj = obj@architecture(train_sample_plain.subsamples(1),{t_pca},classifier,logger);
            obj.kept_energy = kept_energy;
        end
    end
end
