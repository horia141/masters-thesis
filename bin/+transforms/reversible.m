classdef reversible < transform    
    methods (Access=public)
        function [obj] = reversible(logger)
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);

            obj = obj@transform(logger);            
        end
        
        function [dataset_plain_hat] = decode(obj,dataset_coded,logger)
            assert(tc.scalar(obj));
            assert(tc.transforms_reversible(obj));
            assert(tc.scalar(dataset_coded));
            assert(tc.dataset(dataset_coded));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(obj.one_sample_coded.compatible(dataset_coded));
            
            dataset_plain_hat = obj.do_decode(dataset_coded,logger);
        end
    end
    
    methods (Abstract,Access=protected)
        do_decode(dataset_coded,logger);
    end
end
