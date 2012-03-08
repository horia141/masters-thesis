classdef zca_transform < reversible_transform
    properties (GetAccess=public,SetAccess=immutable)
        coeffs;
        coeffs_eigenvalues;
        samples_mean;
        features_count;
        div_epsilon;
    end
    
    methods (Access=public)
        function [obj] = zca_transform(samples,div_epsilon)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(~exist('div_epsilon','var') || ...
                   (tc.scalar(div_epsilon) && tc.number(div_epsilon) && tc.check(div_epsilon >= 0)));
            
            if exist('div_epsilon','var')
                div_epsilon_t = div_epsilon;
            else
                div_epsilon_t = 0;
            end
            
            [coeffs_t,~,coeffs_eigenvalues_t] = princomp(samples.samples);
            
            obj.coeffs = coeffs_t;
            obj.coeffs_eigenvalues = coeffs_eigenvalues_t;
            obj.samples_mean = mean(samples.samples,1);
            obj.features_count = samples.features_count;
            obj.div_epsilon = div_epsilon_t;
        end
        
        function [new_samples] = code(obj,samples)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(obj.features_count == samples.features_count);
            
            new_samples_t1 = bsxfun(@minus,samples.samples,obj.samples_mean);
            new_samples_t2 = new_samples_t1 * obj.coeffs;
            new_samples_t3 = new_samples_t2 * diag(1 ./ sqrt(obj.coeffs_eigenvalues + obj.div_epsilon));
            new_samples_t4 = new_samples_t3 * obj.coeffs';
            
            new_samples = samples_set(samples.classes,new_samples_t4,samples.labels_idx);
        end
        
        function [new_samples] = decode(obj,samples)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(obj.features_count == samples.features_count);
            
            new_samples_t1 = samples.samples * obj.coeffs;
            new_samples_t2 = new_samples_t1 * diag(sqrt(obj.coeffs_eigenvalues + obj.div_epsilon));
            new_samples_t3 = new_samples_t2 * obj.coeffs';
            new_samples_t4 = bsxfun(@plus,new_samples_t3,obj.samples_mean);
            
            new_samples = samples_set(samples.classes,new_samples_t4,samples.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "zca_transform".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Without specifing argument "div_epsilon".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,~,p_latent] = princomp(A);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = zca_transform(s);
            
            assert(all(size(t.coeffs) == [2 2]));
            assert(utils.approx(t.coeffs,p_A));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(length(t.coeffs_eigenvalues) == 2);
            assert(utils.approx(t.coeffs_eigenvalues,p_latent));
            assert(length(t.samples_mean) == 2);
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.div_epsilon == 0);
            
            clearvars -except display;
            
            fprintf('    With specifing of argument "div_epsilon".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,~,p_latent] = princomp(A);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = zca_transform(s,1e-5);
            
            assert(all(size(t.coeffs) == [2 2]));
            assert(utils.approx(t.coeffs,p_A));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(length(t.coeffs_eigenvalues) == 2);
            assert(utils.approx(t.coeffs_eigenvalues,p_latent));
            assert(length(t.samples_mean) == 2);
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.div_epsilon == 1e-5);
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,~,p_latent] = princomp(A);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = zca_transform(s);
            
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes(1),'none'));
            assert(s_p.classes_count == 1);
            assert(all(size(s_p.samples) == [100 2]));
            assert(utils.approx(s_p.samples,bsxfun(@minus,A,mean(A,1)) * (p_A * diag(1 ./ sqrt(p_latent)) * p_A')));
            assert(utils.approx(cov(s_p.samples),eye(2,2)));
            assert(length(s_p.labels_idx) == 100);
            assert(all(s_p.labels_idx == c));
            assert(s_p.samples_count == 100);
            assert(s_p.features_count == 2);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                scatter(s.samples(:,1),s.samples(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p.samples(:,1),s_p.samples(:,2),'x');
                axis([-4 6 -4 6]);
                title('ZCA transformed samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = zca_transform(s);
            
            s_p = t.code(s);
            s_r = t.decode(s_p);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes(1),'none'));
            assert(s_p.classes_count == 1);
            assert(all(size(s_p.samples) == [100 2]));
            assert(utils.approx(s_r.samples,s.samples));
            assert(length(s_p.labels_idx) == 100);
            assert(all(s_p.labels_idx == c));
            assert(s_p.samples_count == 100);
            assert(s_p.features_count == 2);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,3,1);
                scatter(s.samples(:,1),s.samples(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,3,2);
                scatter(s_p.samples(:,1),s_p.samples(:,2),'x');
                axis([-4 6 -4 6]);
                title('ZCA transformed samples.');
                subplot(1,3,3);
                hold('on');
                scatter(s.samples(:,1),s.samples(:,2),'o','r');
                scatter(s_r.samples(:,1),s_r.samples(:,2),'.','b');
                axis([-4 6 -4 6]);
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Apply ZCA on image patches.\n');
            
            s1 = gray_images_set.load_from_dir('../data/test/scenes_small');
            t1 = patch_extract_transform(500,16,16,0.01);
            s2 = t1.code(s1);
            t2 = zca_transform(s2,0.95);
            
            s2_p = t2.code(s2);
            
            s3 = gray_images_set.from_samples(s2_p,16,16,'remap');
            
            assert(length(s3.classes) == 1);
            assert(strcmp(s3.classes(1),'none'));
            assert(s3.classes_count == 1);
            assert(all(size(s3.samples) == [500 256]));
            assert(tc.matrix(s3.samples) && tc.unitreal(s3.samples));
            assert(all(size(s3.labels_idx) == [500 1]));
            assert(all(s3.labels_idx == ones(500,1)));
            assert(s3.samples_count == 500);
            assert(s3.features_count == 256);
            assert(tc.tensor(s3.images,3) && tc.unitreal(s3.images));
            assert(s3.row_count == 16);
            assert(s3.col_count == 16);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s2.images(:,:,1:4:end)));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s3.images(:,:,1:4:end)));
                title('Reconstructed images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
