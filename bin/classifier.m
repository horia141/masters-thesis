classdef classifier
    properties (GetAccess=public,SetAccess=immutable)
        one_sample;
    end
    
    methods (Access=public)
        function [obj] = classifier(train_dataset)
            assert(tc.scalar(train_dataset) && tc.dataset(train_dataset));
            assert(train_dataset.samples_count == 1);
            
            obj.one_sample = train_dataset.subsamples(1);
        end
        
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                  score,conf_matrix,misclassified] = classify(obj,dataset_d)
            assert(tc.scalar(dataset_d) && tc.dataset(dataset_d));
            assert(obj.one_sample.compatible(dataset_d));
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = obj.do_classify(dataset_d);
            score = 100 * sum(dataset_d.labels_idx == labels_idx_hat) / length(dataset_d.labels_idx);
            conf_matrix = confusionmat(dataset_d.labels_idx,labels_idx_hat);
            misclassified = find(labels_idx_hat ~= dataset_d.labels_idx);
        end
    end
    
    methods (Abstract,Access=protected)
        do_classify(obj,dataset_d);
    end
end
