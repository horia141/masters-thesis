classdef pca_whitening < transforms.reversible
    properties (GetAccess=public,SetAccess=immutable)
        coeffs;
        coeffs_eigenvalues;
        samples_mean;
        kept_energy;
        coded_features_count;
        div_epsilon;
    end
    
    methods (Access=public)
        function [obj] = pca_whitening(train_dataset_plain,kept_energy,div_epsilon)
            assert(tc.scalar(train_dataset_plain) && tc.dataset(train_dataset_plain));
            assert(train_dataset_plain.samples_count >= 1);
            assert(tc.scalar(kept_energy) && tc.unitreal(kept_energy));
            assert(~exist('div_epsilon','var') || ...
                   (tc.scalar(div_epsilon) && tc.number(div_epsilon) && tc.check(div_epsilon >= 0)));
            
            if exist('div_epsilon','var')
                div_epsilon_t = div_epsilon;
            else
                div_epsilon_t = 0;
            end
            
            samples_mean_t = mean(train_dataset_plain.samples,1);
            [coeffs_t,~,coeffs_eigenvalues_t] = princomp(train_dataset_plain.samples);
            
            energy_per_comp_rel = cumsum(coeffs_eigenvalues_t);
            kept_energy_rel = kept_energy * energy_per_comp_rel(end);
            coded_features_count_t = find(energy_per_comp_rel >= kept_energy_rel,1);
            
            one_sample_samples_t = train_dataset_plain.samples(1,:) - samples_mean_t;
            one_sample_samples_t = one_sample_samples_t * coeffs_t(:,1:coded_features_count_t);
            one_sample_samples_t = one_sample_samples_t * diag(1 ./ sqrt(coeffs_eigenvalues_t(1:coded_features_count_t) + div_epsilon_t));
            one_sample_coded_t = dataset(train_dataset_plain.classes,one_sample_samples_t,train_dataset_plain.labels_idx(1));
            
            obj = obj@transforms.reversible(train_dataset_plain.subsamples(1),one_sample_coded_t);
            obj.coeffs = coeffs_t;
            obj.coeffs_eigenvalues = coeffs_eigenvalues_t;
            obj.samples_mean = samples_mean_t;
            obj.kept_energy = kept_energy;
            obj.coded_features_count = coded_features_count_t;
            obj.div_epsilon = div_epsilon_t;
        end
    end
    
    methods (Access=protected)
        function [dataset_coded] = do_code(obj,dataset_plain)
            samples_coded = bsxfun(@minus,dataset_plain.samples,obj.samples_mean);
            samples_coded = samples_coded * obj.coeffs(:,1:obj.coded_features_count);
            samples_coded = samples_coded * diag(1 ./ sqrt(obj.coeffs_eigenvalues(1:obj.coded_features_count) + obj.div_epsilon));
            
            dataset_coded = dataset(dataset_plain.classes,samples_coded,dataset_plain.labels_idx);
        end
        
        function [dataset_plain_hat] = do_decode(obj,dataset_coded)
            coeffs_t = obj.coeffs';
            samples_plain_hat = dataset_coded.samples * diag(sqrt(obj.coeffs_eigenvalues(1:obj.coded_features_count) + obj.div_epsilon));
            samples_plain_hat = samples_plain_hat * coeffs_t(1:obj.coded_features_count,:);
            samples_plain_hat = bsxfun(@plus,samples_plain_hat,obj.samples_mean);
            
            dataset_plain_hat = dataset(dataset_coded.classes,samples_plain_hat,dataset_coded.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.pca_whitening".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With 90%% kept energy and without specifing argument "div_epsilon".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,A_s,p_latent] = princomp(A);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.pca_whitening(s,0.9);
            
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
            assert(utils.approx(t.one_sample_coded.samples,A_s(1,1) / sqrt(p_latent(1))));
            assert(tc.check(t.one_sample_coded.labels_idx == c(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 1);
            assert(utils.approx(t.coeffs,p_A));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(length(t.coeffs_eigenvalues) == 2);
            assert(utils.approx(t.coeffs_eigenvalues,p_latent));
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.kept_energy == 0.9);
            assert(t.coded_features_count == 1);
            assert(t.div_epsilon == 0);
            
            clearvars -except display;
            
            fprintf('    With 90%% kept energy and with specifing of argument "div_epsilon".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,A_s,p_latent] = princomp(A);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.pca_whitening(s,0.9,1e-5);
            
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
            assert(utils.approx(t.one_sample_coded.samples,A_s(1,1) / sqrt(p_latent(1) + 1e-5)));
            assert(tc.check(t.one_sample_coded.labels_idx == c(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 1);
            assert(utils.approx(t.coeffs,p_A));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(utils.approx(t.coeffs_eigenvalues,p_latent));
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.kept_energy == 0.9);
            assert(t.coded_features_count == 1);
            assert(t.div_epsilon == 1e-5);
            
            clearvars -except display;
            
            fprintf('    With 100%% kept energy.\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,A_s,p_latent] = princomp(A);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.pca_whitening(s,1);
            
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
            assert(utils.approx(t.one_sample_coded.samples,A_s(1,:) * diag(1 ./ sqrt(p_latent))));
            assert(tc.check(t.one_sample_coded.labels_idx == c(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 2);
            assert(utils.approx(t.coeffs,p_A));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(length(t.coeffs_eigenvalues) == 2);
            assert(utils.approx(t.coeffs_eigenvalues,p_latent));
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.kept_energy == 1);
            assert(t.coded_features_count == 2);
            assert(t.div_epsilon == 0);
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With 90%% kept energy.\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [~,A_s,p_latent] = princomp(A);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.pca_whitening(s,0.9);
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(utils.approx(s_p.samples,A_s(:,1) * diag(1 ./ sqrt(p_latent(1)))));
            assert(utils.approx(var(s_p.samples),1));
            assert(tc.check(s_p.labels_idx == c));
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
            [~,A_s,p_latent] = princomp(A);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.pca_whitening(s,1);
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(utils.approx(s_p.samples,A_s * diag(1 ./ sqrt(p_latent))));
            assert(utils.approx(var(s_p.samples),[1 1]));
            assert(utils.approx(cov(s_p.samples),eye(2,2)));
            assert(tc.check(s_p.labels_idx == c));
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
            
            fprintf('    With 90%% kept energy.\n');
            
            A = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            c = ones(100,1);
            s = dataset({'none'},A,c);
            
            t = transforms.pca_whitening(s,0.9);
            s_p = t.code(s);
            s_r = t.decode(s_p);
            
            assert(length(s_r.classes) == 1);
            assert(strcmp(s_r.classes{1},'none'));
            assert(s_r.classes_count == 1);
            assert(tc.check(size(s_r.samples) == [100 2]));
            assert(tc.matrix(s_r.samples) && tc.number(s_r.samples));
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
            
            fprintf('    With 100%% kept energy.\n');
            
            A = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.pca_whitening(s,1);            
            s_p = t.code(s);
            s_r = t.decode(s_p);
            
            assert(length(s_r.classes) == 1);
            assert(strcmp(s_r.classes{1},'none'));
            assert(s_r.classes_count == 1);
            assert(tc.check(size(s_r.samples) == [100 2]));
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
            
            fprintf('  Apply PCA Whitening on image patches.\n');
            
            fprintf('    With 95%% kept energy.\n');
            
            s1 = datasets.image.load_from_dir('../data/test/scenes_small');
            t1 = transforms.image.patch_extract(s1,200,10,10,0.0001);
            s2 = t1.code(s1);
            
            t2 = transforms.pca_whitening(s2,0.95);
            s2_p = t2.code(s2);
            s2_r = t2.decode(s2_p);
            
            s3 = datasets.image.from_dataset(s2_r,1,10,10,'clamp');
            
            assert(length(s3.classes) == 1);
            assert(strcmp(s3.classes{1},'none'));
            assert(s3.classes_count == 1);
            assert(tc.check(size(s3.samples) == [200 100]));
            assert(tc.matrix(s3.samples) && tc.unitreal(s3.samples));
            assert(tc.check(size(s3.labels_idx) == [200 1]));
            assert(tc.check(s3.labels_idx == ones(200,1)));
            assert(s3.samples_count == 200);
            assert(s3.features_count == 100);
            assert(tc.tensor(s3.images,4) && tc.unitreal(s3.images));
            assert(s3.layers_count == 1);
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
            
            s1 = datasets.image.load_from_dir('../data/test/scenes_small');
            t1 = transforms.image.patch_extract(s1,200,10,10,0.0001);
            s2 = t1.code(s1);
            
            t2 = transforms.pca_whitening(s2,1);            
            s2_p = t2.code(s2);
            s2_r = t2.decode(s2_p);
            
            s3 = datasets.image.from_dataset(s2_r,1,10,10,'clamp');
            
            assert(length(s3.classes) == 1);
            assert(strcmp(s3.classes{1},'none'));
            assert(s3.classes_count == 1);
            assert(utils.approx(s3.samples,s2.samples));
            assert(tc.check(s3.labels_idx == ones(200,1)));
            assert(s3.samples_count == 200);
            assert(s3.features_count == 100);
            assert(utils.approx(s3.images,s2.images));
            assert(s3.layers_count == 1);
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
            
            fprintf('    With 95%% kept energy on color images.\n');
            
            s1 = datasets.image.load_from_dir('../data/test/scenes_small','color');
            t1 = transforms.image.patch_extract(s1,200,10,10,0.0001);
            s2 = t1.code(s1);
            
            t2 = transforms.pca_whitening(s2,0.95);
            s2_p = t2.code(s2);
            s2_r = t2.decode(s2_p);
            
            s3 = datasets.image.from_dataset(s2_r,3,10,10,'clamp');
            
            assert(length(s3.classes) == 1);
            assert(strcmp(s3.classes{1},'none'));
            assert(s3.classes_count == 1);
            assert(tc.check(size(s3.samples) == [200 300]));
            assert(tc.matrix(s3.samples) && tc.unitreal(s3.samples));
            assert(tc.check(size(s3.labels_idx) == [200 1]));
            assert(tc.check(s3.labels_idx == ones(200,1)));
            assert(s3.samples_count == 200);
            assert(s3.features_count == 300);
            assert(tc.tensor(s3.images,4) && tc.unitreal(s3.images));
            assert(s3.layers_count == 3);
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
