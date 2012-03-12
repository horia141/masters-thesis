classdef binary_classifier < classifier
    methods (Access=public)
        function [obj] = binary_classifier(train_samples)
            assert(tc.scalar(train_samples) && tc.samples_set(train_samples));
            assert(train_samples.samples_count > 0);
            assert(length(unique(train_samples.labels_idx)) == 2);
            
            obj = obj@classifier(train_samples);
        end
    end
end
