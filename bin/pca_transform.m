classdef pca_transform < reversible_transform
    properties (GetAccess=public,SetAccess=immutable)
        coeffs;
        samples_mean;
        kept_energy;
        input_features_count;
        output_features_count;
    end
    
    methods (Access=public)
        function [obj] = pca_transform(samples,kept_energy)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(tc.scalar(kept_energy) && tc.unitreal(kept_energy));
            
            [coeffs_t,~,latent] = princomp(samples.samples);
            
            energy_per_comp_rel = cumsum(latent);
            kept_energy_rel = kept_energy * energy_per_comp_rel(end);
            output_features_count_t = find(energy_per_comp_rel >= kept_energy_rel,1);
            
            obj.coeffs = coeffs_t;
            obj.samples_mean = mean(samples.samples,1);
            obj.kept_energy = kept_energy;
            obj.input_features_count = samples.features_count;
            obj.output_features_count = output_features_count_t;
        end
        
        function [new_samples] = code(obj,samples)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(obj.input_features_count == samples.features_count);
            
            new_samples_t1 = bsxfun(@minus,samples.samples,obj.samples_mean);
            new_samples_t2 = new_samples_t1 * obj.coeffs(:,1:obj.output_features_count);
            
            new_samples = samples_set(samples.classes,new_samples_t2,samples.labels_idx);
        end
        
        function [new_samples] = decode(obj,samples)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(obj.output_features_count == samples.features_count);
            
            coeffs_t = obj.coeffs';
            new_samples_t1 = samples.samples * coeffs_t(1:obj.output_features_count,:);
            new_samples_t2 = bsxfun(@plus,new_samples_t1,obj.samples_mean);
            
            new_samples = samples_set(samples.classes,new_samples_t2,samples.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "pca_transform".\n');
            
            fprintf('  Proper construction.\n');
            
            A = mvnrnd([3 3],[1 0.6; 0.6 0.5],100);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_transform(s,0.9);
            
            assert(all(size(t.coeffs) == [2 2]));
            assert(utils.approx(t.coeffs,princomp(A)));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(length(t.samples_mean) == 2);
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.kept_energy == 0.9);
            assert(t.input_features_count == 2);
            assert(t.output_features_count == 1);
            
            clearvars -except display;
            
            A = mvnrnd([3 3],[1 0.6; 0.6 0.5],100);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_transform(s,1);
            
            assert(all(size(t.coeffs) == [2 2]));
            assert(utils.approx(t.coeffs,princomp(A)));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(length(t.samples_mean) == 2);
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.kept_energy == 1);
            assert(t.input_features_count == 2);
            assert(t.output_features_count == 2);
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With 90%% kept energy.\n');
            
            A = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            [p_A,~,p_latent] = princomp(A);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_transform(s,0.9);
            
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes(1),'none'));
            assert(s_p.classes_count == 1);
            assert(all(size(s_p.samples) == [100 1]));
            assert(utils.approx(s_p.samples,bsxfun(@minus,A,mean(A,1)) * p_A(:,1)));
            assert(length(s_p.labels_idx) == 100);
            assert(all(s_p.labels_idx == c));
            assert(s_p.samples_count == 100);
            assert(s_p.features_count == 1);
            assert(utils.approx(var(s_p.samples),p_latent(1)));
              
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                scatter(s.samples(:,1),s.samples(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p.samples(:,1),zeros(100,1),'x');
                axis([-4 6 -4 6]);
                title('PCA transformed samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With 100%% kept energy.\n');
            
            A = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            [p_A,~,p_latent] = princomp(A);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_transform(s,1);
            
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes(1),'none'));
            assert(s_p.classes_count == 1);
            assert(all(size(s_p.samples) == [100 2]));
            assert(utils.approx(s_p.samples,bsxfun(@minus,A,mean(A,1)) * p_A));
            assert(length(s_p.labels_idx) == 100);
            assert(all(s_p.labels_idx == c));
            assert(s_p.samples_count == 100);
            assert(s_p.features_count == 2);
            assert(utils.approx(var(s_p.samples),p_latent'));
            assert(utils.approx(cov(s_p.samples),diag(p_latent)));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                scatter(s.samples(:,1),s.samples(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p.samples(:,1),zeros(100,1),'x');
                axis([-4 6 -4 6]);
                title('PCA transformed samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            fprintf('    With 90%% kept energy.\n');
            
            A = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_transform(s,0.9);
            
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
                title('PCA transformed samples.');
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
            
            fprintf('    With 100%% kept energy.\n');
            
            A = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            c = ones(100,1);
            
            s = samples_set({'none'},A,c);
            t = pca_transform(s,1);
            
            s_p = t.code(s);
            s_r = t.decode(s_p);
            
            assert(length(s_r.classes) == 1);
            assert(strcmp(s_r.classes(1),'none'));
            assert(s_r.classes_count == 1);
            assert(all(size(s_r.samples) == [100 2]));
            assert(utils.approx(s_r.samples,s.samples));
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
                title('PCA transformed samples.');
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
            
            fprintf('  Apply PCA on image patches.\n');
            
            fprintf('    With 95%% kept energy.\n');
            
            s1 = gray_images_set.load_from_dir('../data/test');
            t1 = patch_extract_transform(200,10,10,0.0001);
            s2 = t1.code(s1);
            t2 = pca_transform(s2,0.95);
            
            s2_p = t2.code(s2);
            s2_r = t2.decode(s2_p);
            
            s3 = gray_images_set.from_samples(s2_r,10,10,'clamp');
            
            assert(length(s3.classes) == 1);
            assert(strcmp(s3.classes(1),'none'));
            assert(s3.classes_count == 1);
            assert(all(size(s3.samples) == [200 100]));
            assert(tc.matrix(s3.samples) && tc.unitreal(s3.samples));
            assert(all(size(s3.labels_idx) == [200 1]));
            assert(all(s3.labels_idx == ones(200,1)));
            assert(s3.samples_count == 200);
            assert(s3.features_count == 100);
            assert(tc.tensor(s3.images,3) && tc.unitreal(s3.images));
            assert(s3.row_count == 10);
            assert(s3.col_count == 10);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s2.images));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s3.images));
                title('Reconstructed images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With 100%% kept energy.\n');
            
            s1 = gray_images_set.load_from_dir('../data/test');
            t1 = patch_extract_transform(200,10,10,0.0001);
            s2 = t1.code(s1);
            t2 = pca_transform(s2,1);
            
            s2_p = t2.code(s2);
            s2_r = t2.decode(s2_p);
            
            s3 = gray_images_set.from_samples(s2_r,10,10,'clamp');
            
            assert(length(s3.classes) == 1);
            assert(strcmp(s3.classes(1),'none'));
            assert(s3.classes_count == 1);
            assert(all(size(s3.samples) == [200 100]));
            assert(tc.matrix(s3.samples) && tc.unitreal(s3.samples));
            assert(utils.approx(s3.samples,s2.samples));
            assert(all(size(s3.labels_idx) == [200 1]));
            assert(all(s3.labels_idx == ones(200,1)));
            assert(s3.samples_count == 200);
            assert(s3.features_count == 100);
            assert(tc.tensor(s3.images,3) && tc.unitreal(s3.images));
            assert(utils.approx(s3.images,s2.images));
            assert(s3.row_count == 10);
            assert(s3.col_count == 10);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s2.images));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s3.images));
                title('Reconstructed images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
