classdef dc_offset_transform < transform
    methods (Access=public)
        function [obj] = dc_offset_transform()
        end
        
        function [new_samples] = code(obj,samples)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            
            samples_t = bsxfun(@minus,samples.samples,mean(samples.samples,2));
            new_samples = samples_set(samples.classes,samples_t,samples.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "dc_offset_transform".\n');
            
            fprintf('  Proper construction and "code".\n');
            
            A = [mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)'];
            c = ones(4,1);
            
            s = samples_set({'none'},A,c);
            t = dc_offset_transform();
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes(1),'none'));
            assert(s_p.classes_count == 1);
            assert(all(size(s_p.samples) == [4 50]));
            assert(utils.approx(s_p.samples,A - repmat(mean(A,2),1,50)));
            assert(length(s_p.labels_idx) == 4);
            assert(all(s_p.labels_idx == c));
            assert(s_p.samples_count == 4);
            assert(s_p.features_count == 50);
            assert(utils.approx(mean(s_p.samples,2),zeros(4,1)));
            
            if exist('display','var') && (display == true)
                figure();
                for i = 1:4
                    subplot(4,2,(i - 1)*2 + 1);
                    plot(s.samples(i,:));
                    axis([1 50 -5 5]);
                    subplot(4,2,(i - 1)*2 + 2);
                    plot(s_p.samples(i,:));
                    axis([1 50 -5 5]);
                end
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
