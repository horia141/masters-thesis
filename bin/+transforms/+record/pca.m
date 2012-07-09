classdef pca < transform
    properties (GetAccess=public,SetAccess=immutable)
        coeffs;
        sample_mean;
        kept_energy;
        coded_features_count;
    end
    
    methods (Access=public)
        function [obj] = pca(train_sample_plain,kept_energy,logger)
            assert(check.dataset_record(train_sample_plain));
            assert(check.scalar(kept_energy));
            assert(check.unitreal(kept_energy));
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
            assert(logger.active);
            
            logger.message('Computing dataset mean.');
            
            sample_mean_t = mean(train_sample_plain,2);
            
            logger.message('Computing principal components and associated variances.');
            
            [coeffs_t,~,latent] = princomp(train_sample_plain');
            
            logger.message('Determining number of components to keep.');
            
            energy_per_comp_rel = cumsum(latent);
            kept_energy_rel = kept_energy * energy_per_comp_rel(end);
            coded_features_count_t = find(energy_per_comp_rel >= kept_energy_rel,1);
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = coded_features_count_t;
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.coeffs = coeffs_t';
            obj.sample_mean = sample_mean_t;
            obj.kept_energy = kept_energy;
            obj.coded_features_count = coded_features_count_t;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            logger.message('Projecting onto reduced space of principal components.');

            sample_coded_t1 = bsxfun(@minus,sample_plain,obj.sample_mean);
            sample_coded = obj.coeffs(1:obj.coded_features_count,:) * sample_coded_t1;
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.pca".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With 90%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.correlated_cloud();
            s_s = princomp(s');
            
            t = transforms.record.pca(s,0.9,logg);
            
            assert(check.same(t.coeffs,s_s',0.1));
            assert(check.same(t.coeffs * t.coeffs',eye(2),0.1));
            assert(check.same(t.sample_mean,[3;3],0.1));
            assert(t.kept_energy == 0.9);
            assert(t.coded_features_count == 1);
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,1));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With 100%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.correlated_cloud();
            s_s = princomp(s');
            
            t = transforms.record.pca(s,1,logg);
            
            assert(check.same(t.coeffs,s_s'));
            assert(check.same(t.coeffs * t.coeffs',eye(2),0.1));
            assert(check.same(t.sample_mean,[3;3],0.1));
            assert(t.kept_energy == 1);
            assert(t.coded_features_count == 2);
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,2));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With 90%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.correlated_cloud();
            [~,s_s,p_latent] = princomp(s');
            
            t = transforms.record.pca(s,0.9,logg);
            s_p = t.code(s,logg);
            
            assert(check.same(s_p,s_s(:,1)'));
            assert(check.same(mean(s_p,2),0,0.1));
            assert(check.same(var(s_p),p_latent(1),0.1));

            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p(1,:),zeros(1,10000),'x');
                axis([-4 6 -4 6]);
                axis('square');
                title('PCA transformed samples.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With 100%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.correlated_cloud();
            [~,s_s,p_latent] = princomp(s');
            
            t = transforms.record.pca(s,1,logg);            
            s_p = t.code(s,logg);
            
            assert(check.same(s_p,s_s'));
            assert(check.same(mean(s_p,2),[0;0],0.1));
            assert(check.same(var(s_p,0,2),p_latent,0.1));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p(1,:),s_p(2,:),'x');
                axis([-4 6 -4 6]);
                axis('square');
                title('PCA transformed samples.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Apply PCA on image patches.\n');
            
            fprintf('    With 95%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s1 = utils.testing.scenes_small();
            t1 = transforms.image.patch_extract(s1,200,10,10,0.0001,logg);
            s2 = t1.code(s1,logg);
            s3 = dataset.flatten_image(s2);
            
            t2 = transforms.record.pca(s3,0.95,logg);
            s3_p = t2.code(s3,logg);
            coeffs_t = t2.coeffs';
            s3_aaa = coeffs_t(:,1:t2.coded_features_count) * s3_p;
            s3_r = bsxfun(@plus,s3_aaa,t2.sample_mean);           
            
            s4 = dataset.rebuild_image(s3_r,3,10,10);
            
            assert(check.tensor(s4,4));
            assert(check.same(size(s4),[10 10 3 200]));
            assert(check.number(s4));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                utils.display.as_tiles(s2);
                title('Original images.');
                subplot(1,2,2);
                utils.display.as_tiles(s4);
                title('Reconstructed images.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With 100%% kept energy.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s1 = utils.testing.scenes_small();
            t1 = transforms.image.patch_extract(s1,200,10,10,0.0001,logg);
            s2 = t1.code(s1,logg);
            s3 = dataset.flatten_image(s2);
            
            t2 = transforms.record.pca(s3,1,logg);
            s3_p = t2.code(s3,logg);
            coeffs_t = t2.coeffs';
            s3_aaa = coeffs_t(:,1:t2.coded_features_count) * s3_p;
            s3_r = bsxfun(@plus,s3_aaa,t2.sample_mean);     
            
            s4 = dataset.rebuild_image(s3_r,3,10,10);
            
            assert(check.same(s4,s2));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                utils.display.as_tiles(s2);
                title('Original images.');
                subplot(1,2,2);
                utils.display.as_tiles(s4);
                title('Reconstructed images.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
        end
    end
end
