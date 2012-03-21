classdef reversible_transform < transform
    methods (Abstract,Access=public)
        decode(obj,samples);
    end
end
    