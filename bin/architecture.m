classdef architecture
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain
        transforms;
        transforms_count;
        classifier;
    end
    
    methods (Access=public)
        function [obj] = architecture(one_sample_plain,transforms,classifier,logger)
            assert(tc.scalar(one_sample_plain));
            assert(tc.dataset(one_sample_plain));
            assert(one_sample_plain.samples_count == 1);
            assert(tc.vector(transforms));
            assert(tc.cell(transforms));
            assert(tc.checkf(@tc.scalar,transforms));
            assert(tc.checkf(@tc.transform,transforms));
            assert(tc.scalar(classifier));
            assert(tc.classifier(classifier));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(transforms{1}.one_sample_plain.compatible(one_sample_plain));
            assert(tc.check(arrayfun(@(ii)transforms{ii}.one_sample_plain.compatible(transforms{ii-1}.one_sample_coded),2:length(transforms)),true));
            assert(classifier.one_sample.compatible(transforms{end}.one_sample_coded));
            
            obj.one_sample_plain = one_sample_plain;
            obj.transforms = transforms;
            obj.transforms_count = length(transforms);
            obj.classifier = classifier;
        end
        
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                  score,conf_matrix,misclassified] = classify(obj,dataset_plain,logger)
            assert(tc.scalar(obj));
            assert(tc.architecture(obj));
            assert(tc.scalar(dataset_plain));
            assert(tc.dataset(dataset_plain));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(obj.one_sample_plain.compatible(dataset_plain));
            
            dataset_transformed = cell(obj.transforms_count + 1,1);
            dataset_transformed{1} = dataset_plain;
            
            for i = 1:obj.transforms_count
                dataset_transformed{i+1} = obj.transforms{i}.code(dataset_transformed{i},logger.new_transform('Transform #%d',i));
            end
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
             score,conf_matrix,misclassified] = obj.classifier.classify(dataset_transformed{end},logger.new_classifier('Classifier'));
        end            
    end
end
