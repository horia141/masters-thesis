classdef mean_substract < transforms.reversible
    properties (GetAccess=public,SetAccess=immutable)
        kept_mean;
    end
    
    methods (Access=public)
        function [obj] = mean_substract(train_dataset_plain)
            assert(tc.scalar(train_dataset_plain) && tc.dataset(train_dataset_plain));
            assert(train_dataset_plain.samples_count >= 1);
            
            kept_mean_t = mean(train_dataset_plain.samples,1);
            
            one_sample_coded_samples_t = train_dataset_plain.samples(1,:) - kept_mean_t;
            one_sample_coded_t = dataset(train_dataset_plain.classes,one_sample_coded_samples_t,train_dataset_plain.labels_idx(1));
            
            obj = obj@transforms.reversible(train_dataset_plain.subsamples(1),one_sample_coded_t);
            obj.kept_mean = kept_mean_t;
        end
    end
    
    methods (Access=protected)
        function [dataset_coded] = do_code(obj,dataset_plain)
            samples_coded = bsxfun(@minus,dataset_plain.samples,obj.kept_mean);
            dataset_coded = dataset(dataset_plain.classes,samples_coded,dataset_plain.labels_idx);
        end
        
        function [dataset_plain_hat] = do_decode(obj,dataset_coded)
            samples_plain_hat = bsxfun(@plus,dataset_coded.samples,obj.kept_mean);
            dataset_plain_hat = dataset(dataset_coded.classes,samples_plain_hat,dataset_coded.labels_idx);
        end
    end
        
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.mean_substract".\n');
            
            fprintf('  Proper construction.\n');
            
            A = mvnrnd([3 3],[1 0.4; 0.4 0.3],100);
            c = ones(100,1);
            s = dataset({'none'},A,c);
            
            t = transforms.mean_substract(s);
            
            assert(length(t.one_sample_plain.classes) == 1);
            assert(strcmp(t.one_sample_plain.classes{1},'none'));
            assert(t.one_sample_plain.classes_count == 1);
            assert(tc.check(t.one_sample_plain.samples == A(1,:)));
            assert(tc.check(t.one_sample_plain.labels_idx == c(1)));
            assert(t.one_sample_plain.samples_count == 1);
            assert(t.one_sample_plain.features_count == 2);
            assert(t.one_sample_plain.compatible(s));
            assert(length(t.one_sample_coded.classes) == 1);
            assert(strcmp(t.one_sample_coded.classes{1},'none'));
            assert(t.one_sample_coded.classes_count == 1);
            assert(tc.check(t.one_sample_coded.samples == (A(1,:) - mean(A,1))));
            assert(tc.check(t.one_sample_coded.labels_idx == c(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 2);
            assert(t.one_sample_coded.compatible(s));
            assert(utils.approx(t.kept_mean,mean(A,1)));
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            A = mvnrnd([3 3],[1 0.4; 0.4 0.4],100);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.mean_substract(s);
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(utils.approx(s_p.samples,A - repmat(mean(A,1),100,1)));
            assert(tc.check(s_p.labels_idx == c));
            assert(s_p.samples_count == 100);
            assert(s_p.features_count == 2);
            assert(utils.approx(mean(s_p.samples,1),[0 0]));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                scatter(s.samples(:,1),s.samples(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p.samples(:,1),s_p.samples(:,2),'x');
                axis([-4 6 -4 6]);
                title('Mean substracted samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            A = mvnrnd([3 3],[1 0.4; 0.4 0.4],100);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.mean_substract(s);
            s_p = t.code(s);            
            s_r = t.decode(s_p);
            
            assert(length(s_r.classes) == 1);
            assert(strcmp(s_r.classes{1},'none'));
            assert(s_r.classes_count == 1);
            assert(utils.approx(s_r.samples,A));
            assert(tc.check(s_r.labels_idx == c));
            assert(s_r.samples_count == 100);
            assert(s_r.features_count == 2);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,3,1);
                scatter(s.samples(:,1),s.samples(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,3,2);
                scatter(s_p.samples(:,1),s_p.samples(:,2),'x');
                axis([-4 6 -4 6]);
                title('Mean substracted samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'.');
                axis([-4 6 -4 6]);
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
