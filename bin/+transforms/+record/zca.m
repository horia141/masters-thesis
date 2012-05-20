classdef zca < transforms.reversible
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
            assert(tc.dataset_record(train_sample_plain));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(~exist('div_epsilon','var') || tc.scalar(div_epsilon));
            assert(~exist('div_epsilon','var') || tc.number(div_epsilon));
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
            
            obj = obj@transforms.reversible(input_geometry,output_geometry,logger);
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
        
        function [sample_plain_hat] = do_decode(obj,sample_coded,logger)
            logger.message('Adding saved mean to each sample.');
            logger.message('Projecting onto scaled space of principal components from scaled original space.');
            logger.message('Projecting onto original space from scaled principal components space.');
            
            sample_plain_hat = bsxfun(@plus,obj.saved_transform_decode * sample_coded,obj.sample_mean);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.record.zca".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Without specifing argument "div_epsilon".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[4 2.4; 2.4 2],100)';
            [s_s,~,p_latent] = princomp(s');

            t = transforms.record.zca(s,log);
            
            assert(tc.same(t.saved_transform_code,s_s * diag(1 ./ sqrt(p_latent)) * s_s'));
            assert(tc.same(t.saved_transform_decode,s_s * diag(sqrt(p_latent)) * s_s'));
            assert(tc.same(t.coeffs,s_s'));
            assert(tc.same(t.coeffs * t.coeffs',eye(2)));
            assert(tc.same(t.coeffs_eigenvalues,p_latent));
            assert(tc.same(t.sample_mean,mean(s,2)));
            assert(t.div_epsilon == 0);
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,2));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With specifing of argument "div_epsilon".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[4 2.4; 2.4 2],100)';
            [s_s,~,p_latent] = princomp(s');

            t = transforms.record.zca(s,log,1e-5);
            
            assert(tc.same(t.saved_transform_code,s_s * diag(1 ./ sqrt(p_latent + 1e-5)) * s_s'));
            assert(tc.same(t.saved_transform_decode,s_s * diag(sqrt(p_latent + 1e-5)) * s_s'));
            assert(tc.same(t.coeffs,s_s'));
            assert(tc.same(t.coeffs * t.coeffs',eye(2)));
            assert(tc.same(t.coeffs_eigenvalues,p_latent));
            assert(tc.same(t.sample_mean,mean(s,2)));
            assert(t.div_epsilon == 1e-5);
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,2));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[4 2.4; 2.4 2],100)';
            [s_s,~,p_latent] = princomp(s');
            
            t = transforms.record.zca(s,log);            
            s_p = t.code(s,log);
            
            assert(tc.same(s_p,s_s * diag(1 ./ sqrt(p_latent)) * s_s' * bsxfun(@minus,s,mean(s,2))));
            assert(tc.same(cov(s_p'),eye(2,2)));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Substracting mean from each sample.\n',...
                                                          'Projecting onto scaled space of principal components.\n',...
                                                          'Projecting onto scaled original space from scaled principal components space.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                title('Original sample.');
                subplot(1,2,2);
                scatter(s_p(1,:),s_p(2,:),'x');
                axis([-4 6 -4 6]);
                title('zca transformed sample.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[4 2.4; 2.4 2],100)';
            
            t = transforms.record.zca(s,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.same(s_r,s));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Substracting mean from each sample.\n',...
                                                          'Projecting onto scaled space of principal components.\n',...
                                                          'Projecting onto scaled original space from scaled principal components space.\n',...
                                                          'Adding saved mean to each sample.\n',...
                                                          'Projecting onto scaled space of principal components from scaled original space.\n',...
                                                          'Projecting onto original space from scaled principal components space.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,3,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                title('Original sample.');
                subplot(1,3,2);
                scatter(s_p(1,:),s_p(2,:),'x');
                axis([-4 6 -4 6]);
                title('ZCA transformed sample.');
                subplot(1,3,3);
                hold('on');
                scatter(s(1,:),s(2,:),'o','r');
                scatter(s_r(1,:),s_r(2,:),'.','b');
                axis([-4 6 -4 6]);
                title('Restored sample.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Apply ZCA on image patches.\n');
            
            fprintf('    On grayscale images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s1 = dataset.load_image_from_dir('../test/scenes_small');
            t1 = transforms.image.patch_extract(s1,1500,16,16,0.01,log);
            s2 = t1.code(s1,log);
            s3 = dataset.flatten_image(s2);
            
            t2 = transforms.record.zca(s3,log);            
            s3_p = t2.code(s3,log);
            
            s4 = dataset.rebuild_image(s3_p,1,16,16);
            s5 = utils.remap_images_to_unit(s4,'global');
            
            assert(tc.tensor(s5,4));
            assert(tc.same(size(s5),[16 16 1 1500]));
            assert(tc.unitreal(s5));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting patches:\n',...
                                                          '  Patches 1 to 150.\n',...
                                                          '  Patches 151 to 300.\n',...
                                                          '  Patches 301 to 450.\n',...
                                                          '  Patches 451 to 600.\n',...
                                                          '  Patches 601 to 750.\n',...
                                                          '  Patches 751 to 900.\n',...
                                                          '  Patches 901 to 1050.\n',...
                                                          '  Patches 1051 to 1200.\n',...
                                                          '  Patches 1201 to 1350.\n',...
                                                          '  Patches 1351 to 1500.\n',...
                                                          'Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Substracting mean from each sample.\n',...
                                                          'Projecting onto scaled space of principal components.\n',...
                                                          'Projecting onto scaled original space from scaled principal components space.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s2(:,:,:,1:20:end)));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s5(:,:,:,1:20:end)));
                title('ZCA transformed images.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    On color images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s1 = dataset.load_image_from_dir('../test/scenes_small','original');
            t1 = transforms.image.patch_extract(s1,1500,16,16,0.01,log);
            s2 = t1.code(s1,log);
            s3 = dataset.flatten_image(s2);
            
            t2 = transforms.record.zca(s3,log);            
            s3_p = t2.code(s3,log);
            
            s4 = dataset.rebuild_image(s3_p,3,16,16);
            s5 = utils.remap_images_to_unit(s4,'global');
            
            assert(tc.tensor(s5,4));
            assert(tc.same(size(s5),[16 16 3 1500]));
            assert(tc.unitreal(s5));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting patches:\n',...
                                                          '  Patches 1 to 150.\n',...
                                                          '  Patches 151 to 300.\n',...
                                                          '  Patches 301 to 450.\n',...
                                                          '  Patches 451 to 600.\n',...
                                                          '  Patches 601 to 750.\n',...
                                                          '  Patches 751 to 900.\n',...
                                                          '  Patches 901 to 1050.\n',...
                                                          '  Patches 1051 to 1200.\n',...
                                                          '  Patches 1201 to 1350.\n',...
                                                          '  Patches 1351 to 1500.\n',...
                                                          'Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Substracting mean from each sample.\n',...
                                                          'Projecting onto scaled space of principal components.\n',...
                                                          'Projecting onto scaled original space from scaled principal components space.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s2(:,:,:,1:20:end)));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s5(:,:,:,1:20:end)));
                title('ZCA transformed images.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
