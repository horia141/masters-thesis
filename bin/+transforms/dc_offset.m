classdef dc_offset < transform
    methods (Access=public)
        function [obj] = dc_offset(train_dataset_plain)
            assert(tc.scalar(train_dataset_plain) && tc.dataset(train_dataset_plain));
            assert(train_dataset_plain.samples_count >= 1);
            
            obj = obj@transform(train_dataset_plain.subsamples(1));
        end
    end
    
    methods (Access=protected)        
        function [dataset_coded] = do_code(~,dataset_plain)
            samples_coded = bsxfun(@minus,dataset_plain.samples,mean(dataset_plain.samples,2));
            dataset_coded = dataset(dataset_plain.classes,samples_coded,dataset_plain.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.dc_offset".\n');
            
            fprintf('  Proper construction.\n');
            
            A = [mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)'];
            c = ones(4,1);
            s = dataset({'none'},A,c);
            
            t = transforms.dc_offset(s);
            
            assert(length(t.one_sample_plain.classes) == 1);
            assert(strcmp(t.one_sample_plain.classes{1},'none'));
            assert(t.one_sample_plain.classes_count == 1);
            assert(tc.check(t.one_sample_plain.samples == A(1,:)));
            assert(tc.check(t.one_sample_plain.labels_idx == c(1)));
            assert(t.one_sample_plain.samples_count == 1);
            assert(t.one_sample_plain.features_count == 50);
            assert(t.one_sample_plain.compatible(s));
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            A = [mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)'];
            c = ones(4,1);
            s = dataset({'none'},A,c);
            
            t = transforms.dc_offset(s);
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(utils.approx(s_p.samples,A - repmat(mean(A,2),1,50)));
            assert(tc.check(s_p.labels_idx == c));
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
