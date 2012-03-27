classdef architecture
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain
        transforms;
        transforms_count;
        classifier;
    end
    
    methods (Access=public)
        function [obj] = architecture(one_sample_plain,transforms,classifier)
            assert(tc.scalar(one_sample_plain) && tc.dataset(one_sample_plain));
            assert(one_sample_plain.samples_count == 1);
            assert(tc.vector(transforms) && tc.cell(transforms) && ...
                   cellfun(@(c)tc.scalar(c) && tc.transform(c),transforms));
            assert(tc.scalar(classifier) && tc.classifier(classifier));
            assert(transforms{1}.one_sample_plain.compatible(one_sample_plain));
            assert(tc.check(arrayfun(@(i)transforms{i}.one_sample_plain.compatible(transforms{i-1}.one_sample_coded),2:length(transforms)),true));
            assert(classifier.one_sample.compatible(transforms{end}.one_sample_coded));
            
            obj.one_sample_plain = one_sample_plain;
            obj.transforms = transforms;
            obj.transforms_count = length(transforms);
            obj.classifier = classifier;
        end
        
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                  score,conf_matrix,misclassified] = classify(obj,dataset_plain)
            assert(tc.scalar(obj) && tc.architecture(obj));
            assert(tc.scalar(dataset_plain) && tc.dataset(dataset_plain));
            assert(obj.one_sample_plain.compatible(dataset_plain));
            
            dataset_transformed = cell(obj.transforms_count + 1,1);
            dataset_transformed{1} = dataset_plain;
            
            for i = 1:obj.transforms_count
                dataset_transformed{i+1} = obj.transforms{i}.code(dataset_transformed{i});
            end
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
             score,conf_matrix,misclassified] = obj.classifier.classify(dataset_transformed{end});
        end            
    end
end
