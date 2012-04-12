classdef classifier
    properties (GetAccess=public,SetAccess=immutable)
        one_sample;
    end
    
    methods (Access=public)
        function [obj] = classifier(one_sample,logger)
            assert(tc.scalar(one_sample));
            assert(tc.dataset(one_sample));
            assert(one_sample.samples_count == 1);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            obj.one_sample = one_sample;
        end
        
        function [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2,...
                  score,conf_matrix,misclassified] = classify(obj,dataset_d,logger)
            assert(tc.scalar(dataset_d));
            assert(tc.dataset(dataset_d));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(obj.one_sample.compatible(dataset_d));
            
            [labels_idx_hat,labels_confidence,labels_idx_hat2,labels_confidence2] = obj.do_classify(dataset_d,logger);
            score = 100 * sum(dataset_d.labels_idx == labels_idx_hat) / length(dataset_d.labels_idx);
            conf_matrix = confusionmat(dataset_d.labels_idx,labels_idx_hat);
            misclassified = find(labels_idx_hat ~= dataset_d.labels_idx);
        end
    end
    
    methods (Abstract,Access=protected)
        do_classify(obj,dataset_d);
    end
end
