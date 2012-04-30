classdef reversible < transform    
    methods (Access=public)
        function [obj] = reversible(input_geometry,output_geometry,logger)
            assert(tc.vector(input_geometry));
            assert((length(input_geometry) == 1) || (length(input_geometry) == 4));
            assert(tc.natural(input_geometry));
            assert(tc.check(input_geometry >= 1));
            assert(tc.vector(output_geometry));
            assert((length(output_geometry) == 1) || (length(output_geometry) == 4));
            assert(tc.natural(output_geometry));
            assert(tc.check(output_geometry >= 1));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);

            obj = obj@transform(input_geometry,output_geometry,logger);
        end
        
        function [sample_plain_hat] = decode(obj,sample_coded,logger)
            assert(tc.scalar(obj));
            assert(tc.transforms_reversible(obj));
            assert(tc.dataset(sample_coded));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(dataset.geom_compatible(obj.output_geometry,dataset.geometry(sample_coded)));
            
            sample_plain_hat = obj.do_decode(sample_coded,logger);
        end
    end
    
    methods (Abstract,Access=protected)
        do_decode(sample_coded,logger);
    end
end
