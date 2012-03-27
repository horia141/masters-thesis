classdef zca < transforms.reversible
    properties (GetAccess=public,SetAccess=immutable)
        coeffs;
        coeffs_eigenvalues;
        samples_mean;
        div_epsilon;
    end
    
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = zca(train_dataset_plain,div_epsilon)
            assert(tc.scalar(train_dataset_plain) && tc.dataset(train_dataset_plain));
            assert(train_dataset_plain.samples_count >= 1);
            assert(~exist('div_epsilon','var') || ...
                   (tc.scalar(div_epsilon) && tc.number(div_epsilon) && tc.check(div_epsilon >= 0)));
            
            if exist('div_epsilon','var')
                div_epsilon_t = div_epsilon;
            else
                div_epsilon_t = 0;
            end
            
            samples_mean_t = mean(train_dataset_plain.samples,1);
            [coeffs_t,~,coeffs_eigenvalues_t] = princomp(train_dataset_plain.samples);
            
            obj = obj@transforms.reversible();
            obj.coeffs = coeffs_t;
            obj.coeffs_eigenvalues = coeffs_eigenvalues_t;
            obj.samples_mean = samples_mean_t;
            obj.div_epsilon = div_epsilon_t;
            obj.one_sample_plain = train_dataset_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain);
        end
    end
    
    methods (Access=protected)
        function [dataset_coded] = do_code(obj,dataset_plain)
            dataset_coded = bsxfun(@minus,dataset_plain.samples,obj.samples_mean);
            dataset_coded = dataset_coded * obj.coeffs;
            dataset_coded = dataset_coded * diag(1 ./ sqrt(obj.coeffs_eigenvalues + obj.div_epsilon));
            dataset_coded = dataset_coded * obj.coeffs';
            
            dataset_coded = dataset(dataset_plain.classes,dataset_coded,dataset_plain.labels_idx);
        end
        
        function [dataset_plain_hat] = do_decode(obj,dataset_coded)
            dataset_plain_hat = dataset_coded.samples * obj.coeffs;
            dataset_plain_hat = dataset_plain_hat * diag(sqrt(obj.coeffs_eigenvalues + obj.div_epsilon));
            dataset_plain_hat = dataset_plain_hat * obj.coeffs';
            dataset_plain_hat = bsxfun(@plus,dataset_plain_hat,obj.samples_mean);
            
            dataset_plain_hat = dataset(dataset_coded.classes,dataset_plain_hat,dataset_coded.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.zca".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Without specifing argument "div_epsilon".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,A_s,p_latent] = princomp(A);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.zca(s);
            
            assert(utils.approx(t.coeffs,p_A));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(utils.approx(t.coeffs_eigenvalues,p_latent));
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.div_epsilon == 0);
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
            assert(utils.approx(t.one_sample_coded.samples,A_s(1,:) * diag(1 ./ sqrt(p_latent)) * p_A'));
            assert(tc.check(t.one_sample_coded.labels_idx == c(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 2);
            
            clearvars -except display;
            
            fprintf('    With specifing of argument "div_epsilon".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,A_s,p_latent] = princomp(A);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.zca(s,1e-5);
            
            assert(utils.approx(t.coeffs,p_A));
            assert(utils.approx(t.coeffs * t.coeffs',eye(2)));
            assert(utils.approx(t.coeffs_eigenvalues,p_latent));
            assert(utils.approx(t.samples_mean,mean(A,1)));
            assert(t.div_epsilon == 1e-5);
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
            assert(utils.approx(t.one_sample_coded.samples,A_s(1,:) * diag(1 ./ sqrt(p_latent + 1e-5)) * p_A'));
            assert(tc.check(t.one_sample_coded.labels_idx == c(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 2);            
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            [p_A,~,p_latent] = princomp(A);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.zca(s);            
            s_p = t.code(s);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(utils.approx(s_p.samples,bsxfun(@minus,A,mean(A,1)) * (p_A * diag(1 ./ sqrt(p_latent)) * p_A')));
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
                title('ZCA transformed samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            A = mvnrnd([3 3],[4 2.4; 2.4 2],100);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.zca(s);            
            s_p = t.code(s);
            s_r = t.decode(s_p);
            
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
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
            
            fprintf('    On grayscale images.\n');
            
            s1 = datasets.image.load_from_dir('../data/test/scenes_small');
            t1 = transforms.image.patch_extract(s1,1500,16,16,0.01);
            s2 = t1.code(s1);
            
            t2 = transforms.zca(s2);            
            s2_p = t2.code(s2);
            
            s3 = datasets.image.from_dataset(s2_p,1,16,16,'remap','global');
            
            assert(length(s3.classes) == 1);
            assert(strcmp(s3.classes{1},'none'));
            assert(s3.classes_count == 1);
            assert(tc.check(size(s3.samples) == [1500 256]));
            assert(tc.matrix(s3.samples) && tc.unitreal(s3.samples));
            assert(tc.check(s3.labels_idx == ones(1500,1)));
            assert(s3.samples_count == 1500);
            assert(s3.features_count == 256);
            assert(tc.check(size(s3.images) == [16 16 1 1500]));
            assert(tc.tensor(s3.images,4) && tc.unitreal(s3.images));
            assert(s3.layers_count == 1);
            assert(s3.row_count == 16);
            assert(s3.col_count == 16);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s2.images(:,:,:,1:20:end)));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s3.images(:,:,:,1:20:end)));
                title('Reconstructed images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    On color images.\n');
            
            s1 = datasets.image.load_from_dir('../data/test/scenes_small','color');
            t1 = transforms.image.patch_extract(s1,1500,16,16,0.01);
            s2 = t1.code(s1);
            
            t2 = transforms.zca(s2);            
            s2_p = t2.code(s2);
            
            s3 = datasets.image.from_dataset(s2_p,3,16,16,'remap','global');
            
            assert(length(s3.classes) == 1);
            assert(strcmp(s3.classes{1},'none'));
            assert(s3.classes_count == 1);
            assert(tc.check(size(s3.samples) == [1500 3*256]));
            assert(tc.matrix(s3.samples) && tc.unitreal(s3.samples));
            assert(tc.check(s3.labels_idx == ones(1500,1)));
            assert(s3.samples_count == 1500);
            assert(s3.features_count == 3*256);
            assert(tc.check(size(s3.images) == [16 16 3 1500]));
            assert(tc.tensor(s3.images,4) && tc.unitreal(s3.images));
            assert(s3.layers_count == 3);
            assert(s3.row_count == 16);
            assert(s3.col_count == 16);
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s2.images(:,:,:,1:20:end)));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s3.images(:,:,:,1:20:end)));
                title('Reconstructed images.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
