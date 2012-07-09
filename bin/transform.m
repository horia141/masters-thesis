classdef transform
    properties (GetAccess=public,SetAccess=immutable)
        input_geometry;
        output_geometry;
    end
    
    methods (Access=public)
        function [obj] = transform(input_geometry,output_geometry,logger)
            assert(check.vector(input_geometry));
            assert((length(input_geometry) == 1) || (length(input_geometry) == 4));
            assert(check.natural(input_geometry));
            assert(check.checkv(input_geometry >= 1));
            assert(check.vector(output_geometry));
            assert((length(output_geometry) == 1) || (length(output_geometry) == 4));
            assert(check.natural(output_geometry));
            assert(check.checkv(output_geometry >= 1));
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
            assert(logger.active);
            
            obj.input_geometry = input_geometry;
            obj.output_geometry = output_geometry;
        end
        
        function [sample_coded] = code(obj,sample_plain,logger)
            assert(check.scalar(obj));
            assert(check.transform(obj));
            assert(check.dataset(sample_plain));
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
            assert(logger.active);
            assert(dataset.geom_compatible(obj.input_geometry,dataset.geometry(sample_plain)));
            
            sample_coded = obj.do_code(sample_plain,logger);
        end
    end
    
    methods (Abstract,Access=protected)
        do_code(sample_plain,logger);
    end
end
