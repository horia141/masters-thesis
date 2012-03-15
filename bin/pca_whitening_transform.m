classdef pca_whitening_transform < reversible_transform
    properties (GetAccess=public,SetAccess=immutable)
        coeffs;
        coeffs_eigenvalues;
        samples_mean;
        kept_energy;
        input_features_count;
        output_features_count;
        div_epsilon;
    end
    
    methods (Access=public)
        function [obj] = pca_whitening_transform(samples,kept_energy,div_epsilon)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(tc.scalar(kept_energy) && tc.unitreal(kept_energy));
            assert(~exist('div_epsilon','var') || ...
                   (tc.scalar(div_epsilon) && tc.number(div_epsilon) && tc.check(div_epsilon >= 0)));
            
            if exist('div_epsilon','var')
                div_epsilon_t = div_epsilon;
            else
                div_epsilon_t = 0;
            end
            
            [coeffs_t,~,coeffs_eigenvalues_t] = princomp(samples.samples);
            
            energy_per_comp_rel = cumsum(coeffs_eigenvalues_t);
            kept_energy_rel = kept_energy * energy_per_comp_rel(end);
            output_features_count_t = find(energy_per_comp_rel >= kept_energy_rel,1);
            
            obj.coeffs = coeffs_t;
            obj.coeffs_eigenvalues = coeffs_eigenvalues_t;
            obj.samples_mean = mean(samples.samples,1);
            obj.kept_energy = kept_energy;
            obj.input_features_count = samples.features_count;
            obj.output_features_count = output_features_count_t;
            obj.div_epsilon = div_epsilon_t;
        end
        
        function [new_samples] = code(obj,samples)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(obj.input_features_count == samples.features_count);
            
            new_samples_t1 = bsxfun(@minus,samples.samples,obj.samples_mean);
            new_samples_t2 = new_samples_t1 * obj.coeffs(:,1:obj.output_features_count);
            new_samples_t3 = new_samples_t2 * diag(1 ./ sqrt(obj.coeffs_eigenvalues(1:obj.output_features_count) + obj.div_epsilon));
            
            new_samples = samples_set(samples.classes,new_samples_t3,samples.labels_idx);
        end
        
        function [new_samples] = decode(obj,samples)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(obj.output_features_count == samples.features_count);
            
            coeffs_t = obj.coeffs';
            new_samples_t1 = samples.samples * diag(sqrt(obj.coeffs_eigenvalues(1:obj.output_features_count) + obj.div_epsilon));
            new_samples_t2 = new_samples_t1 * coeffs_t(1:obj.output_features_count,:);
            new_samples_t3 = bsxfun(@plus,new_samples_t2,obj.samples_mean);
            
            new_samples = samples_set(samples.classes,new_samples_t3,samples.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "pca_whitening_transform".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Without specifing argument "div_epsilon".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,~,p_latent] = princomp(A);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_whitening_transform(s,0.9);
            
            assert(all(size(t.coeffs) == [2 2]));
            assert(utils.approx(t.coeffs,p_A));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(length(t.coeffs_eigenvalues) == 2);
            assert(utils.approx(t.coeffs_eigenvalues,p_latent));
            assert(length(t.samples_mean) == 2);
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.kept_energy == 0.9);
            assert(t.input_features_count == 2);
            assert(t.output_features_count == 1);
            assert(t.div_epsilon == 0);
            
            clearvars -except display;
            
            fprintf('    With specifing of argument "div_epsilon".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,~,p_latent] = princomp(A);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_whitening_transform(s,0.9,1e-5);
            
            assert(all(size(t.coeffs) == [2 2]));
            assert(utils.approx(t.coeffs,p_A));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(length(t.coeffs_eigenvalues) == 2);
            assert(utils.approx(t.coeffs_eigenvalues,p_latent));
            assert(length(t.samples_mean) == 2);
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.kept_energy == 0.9);
            assert(t.input_features_count == 2);
            assert(t.output_features_count == 1);
            assert(t.div_epsilon == 1e-5);
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With 90%% kept energy.\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_whitening_transform(s,0.9);
            
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes(1),'none'));
            assert(s_p.classes_count == 1);
            assert(all(size(s_p.samples) == [100 1]));
            assert(utils.approx(var(s_p.samples),1));
            assert(length(s_p.labels_idx) == 100);
            assert(all(s_p.labels_idx == c));
            assert(s_p.samples_count == 100);
            assert(s_p.features_count == 1);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                scatter(s.samples(:,1),s.samples(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p.samples(:,1),zeros(100,1),'x');
                axis([-4 6 -4 6]);
                title('PCA transformed and whitened samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With 100%% kept energy.\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_whitening_transform(s,1);
            
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes(1),'none'));
            assert(s_p.classes_count == 1);
            assert(all(size(s_p.samples) == [100 2]));
            assert(utils.approx(var(s_p.samples),[1 1]));
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
                title('PCA transformed and whitened samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            A = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_whitening_transform(s,0.9);
            
            s_p = t.code(s);
            s_r = t.decode(s_p);
            
            assert(length(s_r.classes) == 1);
            assert(strcmp(s_r.classes(1),'none'));
            assert(s_r.classes_count == 1);
            assert(all(size(s_r.samples) == [100 2]));
            assert(length(s_r.labels_idx) == 100);
            assert(all(s_r.labels_idx == c));
            assert(s_r.samples_count == 100);
            assert(s_r.features_count == 2);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,3,1);
                scatter(s.samples(:,1),s.samples(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,3,2);
                scatter(s_p.samples(:,1),zeros(100,1),'x');
                axis([-4 6 -4 6]);
                title('PCA transformed and whitened samples.');
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
        end
    end
end
