classdef dc_offset_transform < transform
    methods (Access=public)
        function [obj] = dc_offset_transform()
        end
        
        function [new_samples] = code(obj,samples)
            assert(isa(samples,'samples_set'));
            
            samples_t = bsxfun(@minus,samples.samples,mean(samples.samples,2));
            new_samples = samples_set(samples.classes,samples_t,samples.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test
            fprintf('Testing "dc_offset_transform".\n');
            
            fprintf('  Testing propert construction and "code".\n');
            
            A = [mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)'];
            c = [1 1 2 2];
            
            s = samples_set({'1' '2'},A,c);
            t = dc_offset_transform();
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 2);
            assert(strcmp(s_p.classes(1),'1'));
            assert(strcmp(s_p.classes(2),'2'));
            assert(s_p.classes_count == 2);
            assert(all(size(s_p.samples) == [4 50]));
            assert(all(all(s_p.samples == (A - repmat(mean(A,2),1,50)))));
            assert(length(s_p.labels_idx) == 4);
            assert(all(s_p.labels_idx == c'));
            assert(s_p.samples_count == 4);
            assert(s_p.features_count == 50);
            assert(all((mean(s_p.samples,2) - [0;0;0;0]) < 1e-7));
            
            h = figure();
            
            for i = 1:4
                ax = subplot(4,2,(i - 1)*2 + 1,'Parent',h);
                plot(ax,s.samples(i,:));
                axis(ax,[1 50 -5 5]);
                ax = subplot(4,2,(i - 1)*2 + 2,'Parent',h);
                plot(ax,s_p.samples(i,:));
                axis(ax,[1 50 -5 5]);
            end
            
            pause(5);
            
            close(h);
            clear all;
        end
    end
end
