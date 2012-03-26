classdef binary < classifier
    methods (Access=public)
        function [obj] = binary(train_dataset)
            assert(tc.scalar(train_dataset) && tc.dataset(train_dataset));
            assert(train_dataset.samples_count >= 1);
            assert(length(unique(train_dataset.labels_idx)) == 2);
            
            obj = obj@classifier(train_dataset.subsamples(1));
        end
    end
end
