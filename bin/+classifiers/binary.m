classdef binary < classifier
    methods (Access=public)
        function [obj] = binary(two_samples,logger)
            assert(tc.scalar(two_samples));
            assert(tc.dataset(two_samples));
            assert(two_samples.samples_count == 2);
            assert(length(unique(two_samples.labels_idx)) == 2);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            obj = obj@classifier(two_samples.subsamples(1),logger);
        end
    end
end
