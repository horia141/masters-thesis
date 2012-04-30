classdef binary < classifier
    methods (Access=public)
        function [obj] = binary(input_geometry,saved_labels,logger)
            assert(tc.vector(input_geometry));
            assert((length(input_geometry) == 1) || (length(input_geometry) == 4));
            assert(tc.natural(input_geometry));
            assert(tc.check(input_geometry >= 1));
            assert(tc.vector(saved_labels));
            assert(length(saved_labels) == 2);
            assert(tc.labels(saved_labels));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);

            obj = obj@classifier(input_geometry,saved_labels,logger);
        end
    end
end
