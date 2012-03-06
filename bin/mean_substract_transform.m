classdef mean_substract_transform < reversible_transform
    properties (GetAccess=public,SetAccess=immutable)
        kept_mean;
    end
    
    methods (Access=public)
        function [obj] = mean_substract_transform(samples)
            assert(isa(samples,'samples_set'));
            
            obj.kept_mean = mean(samples.samples,1);
        end
        
        function [new_samples] = code(obj,samples)
            assert(isa(samples,'samples_set'));
            assert(tc.match_dims(obj.kept_mean,samples.samples,2,2));
            
            samples_t = bsxfun(@minus,samples.samples,obj.kept_mean);
            new_samples = samples_set(samples.classes,samples_t,samples.labels_idx);
        end
        
        function [new_samples] = decode(obj,samples)
            assert(isa(samples,'samples_set'));
            assert(tc.match_dims(obj.kept_mean,samples.samples,2,2));
            
            samples_t = bsxfun(@plus,samples.samples,obj.kept_mean);
            new_samples = samples_set(samples.classes,samples_t,samples.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test
            fprintf('Testing "mean_substract_transform".\n');
            
            fprintf('  Testing proper construction.\n');
            
            A = mvnrnd([3 3],[1 0.4; 0.4 0.3],100);
            c = randi(2,100,1);
            
            s = samples_set({'1' '2'},A,c);
            t = mean_substract_transform(s);
            
            assert(length(t.kept_mean) == 2);
            assert(all(t.kept_mean == mean(A,1)));
            
            clear all
            
            fprintf('  Testing function "code".\n');
            
            A = mvnrnd([3 3],[1 0.4; 0.4 0.4],100);
            c = randi(2,100,1);
            
            s = samples_set({'1' '2'},A,c);
            t = mean_substract_transform(s);
            
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 2);
            assert(strcmp(s_p.classes(1),'1'));
            assert(strcmp(s_p.classes(2),'2'));
            assert(s_p.classes_count == 2);
            assert(all(size(s_p.samples) == [100 2]));
            assert(all(all(s_p.samples == (A - repmat(mean(A,1),100,1)))));
            assert(length(s_p.labels_idx) == 100);
            assert(all(s_p.labels_idx == c));
            assert(s_p.samples_count == 100);
            assert(s_p.features_count == 2);
            assert(all((mean(s_p.samples,1) - [0 0]) < 1e-7));
            
            h = figure();
            
            ax = subplot(1,2,1);
            scatter(ax,s.samples(:,1),s.samples(:,2),'o');
            axis(ax,[-5 5 -4 6]);
            title(ax,'Original samples.');
            ax = subplot(1,2,2);
            scatter(ax,s_p.samples(:,1),s_p.samples(:,2),'x');
            axis(ax,[-5 5 -4 6]);
            title(ax,'Mean substracted samples.');
            
            pause(5);
            
            close(h);
            clear all;
            
            fprintf('  Testing function "decode".\n');
            
            A = mvnrnd([3 3],[1 0.4; 0.4 0.4],100);
            c = randi(2,100,1);
            
            s = samples_set({'1' '2'},A,c);
            t = mean_substract_transform(s);
            
            s_p = t.code(s);            
            s_r = t.decode(s_p);
            
            assert(length(s_r.classes) == 2);
            assert(strcmp(s_r.classes(1),'1'));
            assert(strcmp(s_r.classes(2),'2'));
            assert(s_r.classes_count == 2);
            assert(all(size(s_r.samples) == [100 2]));
            assert(all(all(s_r.samples == A)));
            assert(length(s_r.labels_idx) == 100);
            assert(all(s_r.labels_idx == c));
            assert(s_r.samples_count == 100);
            assert(s_r.features_count == 2);
            
            h = figure();
            
            ax = subplot(1,3,1);
            scatter(ax,s.samples(:,1),s.samples(:,2),'o');
            axis(ax,[-5 5 -4 6]);
            title(ax,'Original samples.');
            ax = subplot(1,3,2);
            scatter(ax,s_p.samples(:,1),s_p.samples(:,2),'x');
            axis(ax,[-5 5 -4 6]);
            title(ax,'Mean substracted samples.');
            ax = subplot(1,3,3);
            scatter(ax,s_r.samples(:,1),s_r.samples(:,2),'.');
            axis(ax,[-5 5 -4 6]);
            title(ax,'Restored samples.');
            
            pause(5);
            
            close(h);
            clear all;
        end
    end
end
