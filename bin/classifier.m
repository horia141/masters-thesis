classdef classifier
    properties (GetAccess=public,SetAccess=immutable)
        one_sample;
    end
    
    methods (Access=public)
        function [obj] = classifier(train_samples)
            assert(tc.scalar(train_samples) && tc.samples_set(train_samples));
            assert(train_samples.samples_count > 0);
            
            obj.one_sample = train_samples.subsamples(1);
        end
        
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                  score,conf_matrix,misclassified] = classify(obj,samples)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(obj.one_sample.compatible(samples));
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = obj.do_classify(samples);
            score = 100 * sum(samples.labels_idx == labels_idx_hat) / length(samples.labels_idx);
            conf_matrix = confusionmat(samples.labels_idx,labels_idx_hat);
            misclassified = find(labels_idx_hat ~= samples.labels_idx);
        end
    end
    
    methods (Abstract,Access=protected)
        do_classify(samples);
    end
end
