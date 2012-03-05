classdef reversible_transform < transform
    methods (Abstract,Access=public)
        decode(samples);
    end
end
    