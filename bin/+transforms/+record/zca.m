classdef zca < transform
    properties (GetAccess=public,SetAccess=immutable)
        saved_transform_code;
        saved_transform_decode;
        coeffs;
        coeffs_eigenvalues;
        sample_mean;
        div_epsilon;
    end
    
    methods (Access=public)
        function [obj] = zca(train_sample_plain,logger,div_epsilon)
            assert(check.dataset_record(train_sample_plain));
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
            assert(logger.active);
            assert(~exist('div_epsilon','var') || check.scalar(div_epsilon));
            assert(~exist('div_epsilon','var') || check.number(div_epsilon));
            assert(~exist('div_epsilon','var') || (div_epsilon >= 0));
            
            if ~exist('div_epsilon','var')
                div_epsilon = 0;
            end
            
            logger.message('Computing dataset mean.');
            
            sample_mean_t = mean(train_sample_plain,2);
            
            logger.message('Computing principal components and associated variances.');

            [coeffs_t,~,coeffs_eigenvalues_t] = princomp(train_sample_plain');
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = input_geometry;
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.saved_transform_code = coeffs_t * diag(1 ./ sqrt(coeffs_eigenvalues_t + div_epsilon)) * coeffs_t';
            obj.saved_transform_decode = coeffs_t * diag(sqrt(coeffs_eigenvalues_t + div_epsilon)) * coeffs_t';
            obj.coeffs = coeffs_t';
            obj.coeffs_eigenvalues = coeffs_eigenvalues_t;
            obj.sample_mean = sample_mean_t;
            obj.div_epsilon = div_epsilon;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            logger.message('Substracting mean from each sample.');
            logger.message('Projecting onto scaled space of principal components.');
            logger.message('Projecting onto scaled original space from scaled principal components space.');
            
            sample_coded = obj.saved_transform_code * bsxfun(@minus,sample_plain,obj.sample_mean);
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.zca".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Without specifing argument "div_epsilon".\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.correlated_cloud();
            [s_s,~,p_latent] = princomp(s');

            t = transforms.record.zca(s,logg);
            
            assert(check.same(t.saved_transform_code,s_s * diag(1 ./ sqrt(p_latent)) * s_s'));
            assert(check.same(t.saved_transform_decode,s_s * diag(sqrt(p_latent)) * s_s'));
            assert(check.same(t.coeffs,s_s',0.1));
            assert(check.same(t.coeffs * t.coeffs',eye(2),0.1));
            assert(check.same(t.coeffs_eigenvalues,p_latent,0.1));
            assert(check.same(t.sample_mean,[3;3],0.1));
            assert(t.div_epsilon == 0);
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,2));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With specifing of argument "div_epsilon".\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.correlated_cloud();
            [s_s,~,p_latent] = princomp(s');

            t = transforms.record.zca(s,logg,1e-5);
            
            assert(check.same(t.saved_transform_code,s_s * diag(1 ./ sqrt(p_latent + 1e-5)) * s_s'));
            assert(check.same(t.saved_transform_decode,s_s * diag(sqrt(p_latent + 1e-5)) * s_s'));
            assert(check.same(t.coeffs,s_s',0.1));
            assert(check.same(t.coeffs * t.coeffs',eye(2),0.1));
            assert(check.same(t.coeffs_eigenvalues,p_latent,0.1));
            assert(check.same(t.sample_mean,[3;3],0.1));
            assert(t.div_epsilon == 1e-5);
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,2));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.correlated_cloud();
            [s_s,~,p_latent] = princomp(s');
            
            t = transforms.record.zca(s,logg);            
            s_p = t.code(s,logg);
            
            assert(check.same(s_p,s_s * diag(1 ./ sqrt(p_latent)) * s_s' * bsxfun(@minus,s,mean(s,2))));
            assert(check.same(mean(s_p,2),[0;0],0.1));
            assert(check.same(cov(s_p'),eye(2,2)));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original sample.');
                subplot(1,2,2);
                scatter(s_p(1,:),s_p(2,:),'x');
                axis([-4 6 -4 6]);
                axis('square');
                title('zca transformed sample.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Apply ZCA on image patches.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s1 = utils.testing.scenes_small();
            t1 = transforms.image.patch_extract(s1,1500,16,16,0.01,logg);
            s2 = t1.code(s1,logg);
            s3 = dataset.flatten_image(s2);
            
            t2 = transforms.record.zca(s3,logg);            
            s3_p = t2.code(s3,logg);
            
            s4 = dataset.rebuild_image(s3_p,3,16,16);
            s5 = utils.common.remap_images_to_unit(s4);
            
            assert(check.tensor(s5,4));
            assert(check.same(size(s5),[16 16 3 1500]));
            assert(check.unitreal(s5));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                utils.display.as_tiles(s2(:,:,:,1:20:end));
                title('Original images.');
                subplot(1,2,2);
                utils.display.as_tiles(s5(:,:,:,1:20:end));
                title('ZCA transformed images.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
        end
    end
end
