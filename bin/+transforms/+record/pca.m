classdef pca < transforms.reversible
    properties (GetAccess=public,SetAccess=immutable)
        coeffs;
        sample_mean;
        kept_energy;
        coded_features_count;
    end
    
    methods (Access=public)
        function [obj] = pca(train_sample_plain,kept_energy,logger)
            assert(tc.dataset_record(train_sample_plain));
            assert(tc.scalar(kept_energy));
            assert(tc.unitreal(kept_energy));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            logger.message('Computing dataset mean.');
            
            sample_mean_t = mean(train_sample_plain,1);
            
            logger.message('Computing principal components and associated variances.');
            
            [coeffs_t,~,latent] = princomp(train_sample_plain);
            
            logger.message('Determining number of components to keep.');
            
            energy_per_comp_rel = cumsum(latent);
            kept_energy_rel = kept_energy * energy_per_comp_rel(end);
            coded_features_count_t = find(energy_per_comp_rel >= kept_energy_rel,1);
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = coded_features_count_t;
            
            obj = obj@transforms.reversible(input_geometry,output_geometry,logger);
            obj.coeffs = coeffs_t;
            obj.sample_mean = sample_mean_t;
            obj.kept_energy = kept_energy;
            obj.coded_features_count = coded_features_count_t;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            logger.message('Projecting onto reduced space of principal components.');
            
            sample_coded_t1 = bsxfun(@minus,sample_plain,obj.sample_mean);
            sample_coded = sample_coded_t1 * obj.coeffs(:,1:obj.coded_features_count);
        end
        
        function [sample_plain_hat] = do_decode(obj,sample_coded,logger)
            logger.message('Projecting onto original space from principal components space.');
            
            coeffs_t = obj.coeffs';
            sample_plain_hat_t1 = sample_coded * coeffs_t(1:obj.coded_features_count,:);
            sample_plain_hat = bsxfun(@plus,sample_plain_hat_t1,obj.sample_mean);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.record.pca".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With 90%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0.6; 0.6 0.5],100);
            s_s = princomp(s);
            
            t = transforms.record.pca(s,0.9,log);
            
            assert(tc.same(t.coeffs,s_s));
            assert(tc.same(t.coeffs * t.coeffs',eye(2)));
            assert(tc.same(t.sample_mean,mean(s,1)));
            assert(t.kept_energy == 0.9);
            assert(t.coded_features_count == 1);
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,1));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Determining number of components to keep.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With 100%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0.6; 0.6 0.5],100);
            s_s = princomp(s);
            
            t = transforms.record.pca(s,1,log);
            
            assert(tc.same(t.coeffs,s_s));
            assert(tc.same(t.coeffs * t.coeffs',eye(2)));
            assert(tc.same(t.sample_mean,mean(s,1)));
            assert(t.kept_energy == 1);
            assert(t.coded_features_count == 2);
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,2));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Determining number of components to keep.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With 90%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            [~,s_s,p_latent] = princomp(s);
            
            t = transforms.record.pca(s,0.9,log);            
            s_p = t.code(s,log);
            
            assert(tc.same(s_p,s_s(:,1)));
            assert(tc.same(var(s_p),p_latent(1)));

            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Determining number of components to keep.\n',...
                                                          'Projecting onto reduced space of principal components.\n'))));
              
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                scatter(s(:,1),s(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p(:,1),zeros(100,1),'x');
                axis([-4 6 -4 6]);
                title('PCA transformed samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With 100%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            [~,s_s,p_latent] = princomp(s);
            
            t = transforms.record.pca(s,1,log);            
            s_p = t.code(s,log);
            
            assert(tc.same(s_p,s_s));
            assert(tc.same(var(s_p),p_latent','Epsilon',1e-4));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Determining number of components to keep.\n',...
                                                          'Projecting onto reduced space of principal components.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                scatter(s(:,1),s(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p(:,1),s_p(:,2),'x');
                axis([-4 6 -4 6]);
                title('PCA transformed samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            fprintf('    With 90%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            
            t = transforms.record.pca(s,0.9,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[100 2]));
            assert(tc.number(s_r));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Determining number of components to keep.\n',...
                                                          'Projecting onto reduced space of principal components.\n',...
                                                          'Projecting onto original space from principal components space.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,3,1);
                scatter(s(:,1),s(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,3,2);
                scatter(s_p(:,1),zeros(100,1),'x');
                axis([-4 6 -4 6]);
                title('PCA transformed samples.');
                subplot(1,3,3);
                hold('on');
                scatter(s(:,1),s(:,2),'o','r');
                scatter(s_r(:,1),s_r(:,2),'.','b');
                axis([-4 6 -4 6]);
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With 100%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = mvnrnd([3 3],[1 0.6; 0.6 0.4],100);
            
            t = transforms.record.pca(s,1,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.same(s_r,s));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Determining number of components to keep.\n',...
                                                          'Projecting onto reduced space of principal components.\n',...
                                                          'Projecting onto original space from principal components space.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,3,1);
                scatter(s(:,1),s(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,3,2);
                scatter(s_p(:,1),s_p(:,2),'x');
                axis([-4 6 -4 6]);
                title('PCA transformed samples.');
                subplot(1,3,3);
                hold('on');
                scatter(s(:,1),s(:,2),'o','r');
                scatter(s_r(:,1),s_r(:,2),'.','b');
                axis([-4 6 -4 6]);
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Apply PCA on image patches.\n');
            
            fprintf('    With 95%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s1 = dataset.load_image_from_dir('../test/scenes_small');
            t1 = transforms.image.patch_extract(s1,200,10,10,0.0001,log);
            s2 = t1.code(s1,log);
            s3 = dataset.flatten_image(s2);
            
            t2 = transforms.record.pca(s3,0.95,log);
            s3_p = t2.code(s3,log);
            s3_r = t2.decode(s3_p,log);
            
            s4 = dataset.rebuild_image(s3_r,1,10,10);
            
            assert(tc.tensor(s4,4));
            assert(tc.same(size(s4),[10 10 1 200]));
            assert(tc.number(s4));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting patches:\n',...
                                                          '  Patches 1 to 20.\n',...
                                                          '  Patches 21 to 40.\n',...
                                                          '  Patches 41 to 60.\n',...
                                                          '  Patches 61 to 80.\n',...
                                                          '  Patches 81 to 100.\n',...
                                                          '  Patches 101 to 120.\n',...
                                                          '  Patches 121 to 140.\n',...
                                                          '  Patches 141 to 160.\n',...
                                                          '  Patches 161 to 180.\n',...
                                                          '  Patches 181 to 200.\n',...
                                                          'Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Determining number of components to keep.\n',...
                                                          'Projecting onto reduced space of principal components.\n',...
                                                          'Projecting onto original space from principal components space.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s2));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s4));
                title('Reconstructed images.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With 100%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s1 = dataset.load_image_from_dir('../test/scenes_small');
            t1 = transforms.image.patch_extract(s1,200,10,10,0.0001,log);
            s2 = t1.code(s1,log);
            s3 = dataset.flatten_image(s2);
            
            t2 = transforms.record.pca(s3,1,log);
            s3_p = t2.code(s3,log);
            s3_r = t2.decode(s3_p,log);
            
            s4 = dataset.rebuild_image(s3_r,1,10,10);
            
            assert(tc.same(s4,s2));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting patches:\n',...
                                                          '  Patches 1 to 20.\n',...
                                                          '  Patches 21 to 40.\n',...
                                                          '  Patches 41 to 60.\n',...
                                                          '  Patches 61 to 80.\n',...
                                                          '  Patches 81 to 100.\n',...
                                                          '  Patches 101 to 120.\n',...
                                                          '  Patches 121 to 140.\n',...
                                                          '  Patches 141 to 160.\n',...
                                                          '  Patches 161 to 180.\n',...
                                                          '  Patches 181 to 200.\n',...
                                                          'Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Determining number of components to keep.\n',...
                                                          'Projecting onto reduced space of principal components.\n',...
                                                          'Projecting onto original space from principal components space.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s2));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s4));
                title('Reconstructed images.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With 95%% kept energy on color images.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s1 = dataset.load_image_from_dir('../test/scenes_small','original');
            t1 = transforms.image.patch_extract(s1,200,10,10,0.0001,log);
            s2 = t1.code(s1,log);
            s3 = dataset.flatten_image(s2);
            
            t2 = transforms.record.pca(s3,0.95,log);
            s3_p = t2.code(s3,log);
            s3_r = t2.decode(s3_p,log);
            
            s4 = dataset.rebuild_image(s3_r,3,10,10);
            
            assert(tc.tensor(s4,4));
            assert(tc.same(size(s4),[10 10 3 200]));
            assert(tc.number(s4));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting patches:\n',...
                                                          '  Patches 1 to 20.\n',...
                                                          '  Patches 21 to 40.\n',...
                                                          '  Patches 41 to 60.\n',...
                                                          '  Patches 61 to 80.\n',...
                                                          '  Patches 81 to 100.\n',...
                                                          '  Patches 101 to 120.\n',...
                                                          '  Patches 121 to 140.\n',...
                                                          '  Patches 141 to 160.\n',...
                                                          '  Patches 161 to 180.\n',...
                                                          '  Patches 181 to 200.\n',...
                                                          'Computing dataset mean.\n',...
                                                          'Computing principal components and associated variances.\n',...
                                                          'Determining number of components to keep.\n',...
                                                          'Projecting onto reduced space of principal components.\n',...
                                                          'Projecting onto original space from principal components space.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s2));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s4));
                title('Reconstructed images.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
