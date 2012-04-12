classdef transform
    properties (Abstract,GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = transform(logger)
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
        end
        
        function [dataset_coded] = code(obj,dataset_plain,logger)
            assert(tc.scalar(obj));
            assert(tc.transform(obj));
            assert(tc.scalar(dataset_plain));
            assert(tc.dataset(dataset_plain));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(obj.one_sample_plain.compatible(dataset_plain));
            
            dataset_coded = obj.do_code(dataset_plain,logger);
        end
    end
    
    methods (Abstract,Access=protected)
        do_code(dataset_plain,logger);
    end
end
