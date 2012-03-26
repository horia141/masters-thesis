classdef reversible < transform
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = reversible(one_sample_plain,one_sample_coded)
            assert(tc.scalar(one_sample_plain) && tc.dataset(one_sample_plain));
            assert(one_sample_plain.samples_count == 1);
            assert(tc.scalar(one_sample_coded) && tc.dataset(one_sample_coded));
            assert(one_sample_coded.samples_count == 1);
            assert(utils.same_classes(one_sample_plain.classes,one_sample_coded.classes));
            assert(tc.check(one_sample_plain.labels_idx == one_sample_coded.labels_idx));
            
            obj = obj@transform(one_sample_plain);
            obj.one_sample_coded = one_sample_coded;
        end
        
        function [dataset_plain_hat] = decode(obj,dataset_coded)
            assert(tc.scalar(obj) && tc.transforms_reversible(obj));
            assert(tc.scalar(dataset_coded) && tc.dataset(dataset_coded));
            assert(obj.one_sample_coded.compatible(dataset_coded));
            
            dataset_plain_hat = obj.do_decode(dataset_coded);
        end
    end
    
    methods (Abstract,Access=protected)
        do_decode(obj,dataset_coded);
    end
end
