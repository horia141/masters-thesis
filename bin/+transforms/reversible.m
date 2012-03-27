classdef reversible < transform    
    methods (Access=public)
        function [obj] = reversible()
            obj = obj@transform();
        end
        
        function [dataset_plain_hat] = decode(obj,dataset_coded)
            assert(tc.scalar(obj) && tc.transforms_reversible(obj));
            assert(tc.scalar(dataset_coded) && tc.dataset(dataset_coded));
            assert(obj.one_sample_coded.compatible(dataset_coded));
            
            dataset_plain_hat = obj.do_decode(dataset_coded);
        end
    end
    
    methods (Abstract,Access=protected)
        do_decode(dataset_coded);
    end
end
