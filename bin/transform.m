classdef transform
    properties (Abstract,GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = transform()
        end
        
        function [dataset_coded] = code(obj,dataset_plain)
            assert(tc.scalar(obj) && tc.transform(obj));
            assert(tc.scalar(dataset_plain) && tc.dataset(dataset_plain));
            assert(obj.one_sample_plain.compatible(dataset_plain));
            
            dataset_coded = obj.do_code(dataset_plain);
        end
    end
    
    methods (Abstract,Access=protected)
        do_code(dataset_plain);
    end
end
