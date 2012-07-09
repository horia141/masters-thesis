classdef dictionary < transform
    properties (GetAccess=public,SetAccess=immutable)
        dict;
        dict_transp;
        word_count;
        coding_fn;
        coding_params_cell;
        coding_method;
        coding_params;
    end

    methods (Access=public)
        function [obj] = dictionary(train_sample_plain,dict,coding_method,coding_params,logger)
            assert(check.dataset_record(train_sample_plain));
            assert(check.matrix(dict));
            assert(size(dict,2) == dataset.geometry(train_sample_plain));
            assert(transforms.record.dictionary.coding_setup_ok(size(dict,1),coding_method,coding_params));
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
            assert(logger.active);
            
            dict_t = transforms.record.dictionary.normalize_dict(dict);
            dict_transp_t = dict_t';
            word_count_t = size(dict,1);
            [coding_fn_t,coding_params_cell_t] = transforms.record.dictionary.coding_setup(word_count_t,coding_method,coding_params);
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = word_count_t;
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.dict = dict_t;
            obj.dict_transp = dict_transp_t;
            obj.word_count = word_count_t;
            obj.coding_fn = coding_fn_t;
            obj.coding_params_cell = coding_params_cell_t;
            obj.coding_method = coding_method;
            obj.coding_params = coding_params;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            logger.message('Building sparse samples.');

            sample_coded = obj.coding_fn(obj.dict,obj.dict_transp,sample_plain,obj.coding_params_cell{:});
        end
    end
    
    methods (Static,Access=protected)
        function [o] = coding_setup_ok(word_count,coding_method,coding_params)
            o = true;
            o = o && check.scalar(coding_method);
            o = o && check.string(coding_method);
                o = o && check.one_of(coding_method,'Corr','CorrOrder','MP','OMP','SparseNet');
            o = o && ((check.same(coding_method,'Corr') && check.empty(coding_params)) || ...
                          (check.same(coding_method,'CorrOrder') && (check.vector(coding_params) && ...
                                                                     length(coding_params) == 3 && ...
                                                                     check.unitreal(coding_params(1)) && ...
                                                                     (coding_params(1) > 0) && ...
                                                                     (coding_params(1) < 1) && ...
                                                                     check.unitreal(coding_params(2)) && ...
                                                                     (coding_params(2) < 1) && ...
                                                                     check.natural(coding_params(3)) && ...
                                                                     (coding_params(3) >= 1))) || ...
                          (check.same(coding_method,'MP') && (check.scalar(coding_params) && ...
                                                              check.natural(coding_params) && ...
                                                              (coding_params > 0) && ...
                                                              (coding_params <= word_count))) || ...
                          (check.same(coding_method,'OMP') && (check.scalar(coding_params) && ...
                                                               check.natural(coding_params) && ...
                                                               (coding_params > 0) && ...
                                                               (coding_params <= word_count))) || ...
                          (check.same(coding_method,'SparseNet') && (check.scalar(coding_params) && ...
                                                                     check.unitreal(coding_params) && ...
                                                                     (coding_params > 0))));
        end

        function [coding_fn,coding_params_cell] = coding_setup(word_count,coding_method,coding_params)
            if check.same(coding_method,'Corr')
                coding_fn = @transforms.record.dictionary.correlation;
                coding_params_cell = {};
            elseif check.same(coding_method,'CorrOrder')
                desired_sparseness = coding_params(1);
                minimum_non_zero = coding_params(2);
                non_zero_count = min(ceil(desired_sparseness * word_count),word_count);
                modulator = [minimum_non_zero .^ ((0:(non_zero_count - 1)) / (non_zero_count - 1)) zeros(1,word_count - non_zero_count)];
                coding_fn = @transforms.record.dictionary.correlation_order;
                coding_params_cell = {modulator coding_params(3)};
            elseif check.same(coding_method,'MP')
                coding_fn = @transforms.record.dictionary.matching_pursuit;
                coding_params_cell = {coding_params(1)};
            elseif check.same(coding_method,'OMP')
                coding_fn = @transforms.record.dictionary.ortho_matching_pursuit;
                coding_params_cell = {coding_params(1)};
            elseif check.same(coding_method,'SparseNet')
                coding_fn = @transforms.record.dictionary.sparse_net;
                coding_params_cell = {coding_params(1)};
            else
                assert(false);
            end
        end

        function [norm_dict] = normalize_dict(dict)
            norm_dict = dict ./ repmat(sqrt(sum(dict .^ 2,2)),1,size(dict,2));
        end

        function [coeffs] = correlation(dict,~,sample)
            coeffs = dict * sample;
        end
        
        function [coeffs] = correlation_order(dict,~,sample,modulator,num_threads)
            similarities = dict * sample;
            coeffs = transforms.record.dictionary_correlation_order(similarities,modulator,num_threads);
        end
        
        function [coeffs] = matching_pursuit(dict,dict_transp,sample,coeffs_count)
            coeffs = spalloc(size(dict,1),dataset.count(sample),coeffs_count * size(sample,2));
            sample_residue = sample;
            
            for k = 1:coeffs_count
                similarities = dict * sample_residue;
                [~,best_match] = max(similarities .^ 2,[],1);
                coeffs = coeffs + sparse(best_match,1:size(sample,2),similarities(sub2ind(size(similarities),best_match,1:size(sample,2))),size(dict,1),size(sample,2));
                sample_residue = sample - dict_transp * coeffs;
            end
        end
        
        function [coeffs] = ortho_matching_pursuit(dict,dict_transp,sample,coeffs_count)
            coeffs = spalloc(size(dict,1),dataset.count(sample),coeffs_count * size(sample,2));
            sample_residue = sample;
            selected = zeros(coeffs_count,dataset.count(sample));
            
            for k = 1:coeffs_count
                similarities = dict * sample_residue;
                [~,selected(k,:)] = max(similarities .^ 2,[],1);
                selected_p = selected(1:k,:);
                for ii = 1:size(sample,2)
                    coeffs(selected_p(:,ii),ii) = (dict(selected_p(:,ii),:) * dict_transp(:,selected_p(:,ii))) \ (dict(selected_p(:,ii),:) * sample(:,ii));
                end
                sample_residue = sample - dict_transp * coeffs;
            end
        end
        
        function [coeffs] = sparse_net(dict,dict_transp,sample,lambda_sigma_ratio)
            N = dataset.count(sample);
            w = size(dict,1);
            coeffs = zeros(w,N);
            
            sigma = std(sample(:));
            lambda = lambda_sigma_ratio * sigma;
            S = @(x)log(1 + x.^2);
            dS = @(x)2 * x ./ (1 + x);
            dict_dict_transp = dict * dict_transp;
            initial_coeffs = utils.common.rand_range(-0.05,0.05,w,N);
            optprop = optimset('GradObj','on','Display','off','LargeScale','off');
            
            for ii = 1:N
                coeffs(:,ii) = fminunc(@(x)transforms.record.dictionary.sparse_net_opt(dict,dict_transp,dict_dict_transp,sample(:,ii),x,S,dS,lambda,sigma),initial_coeffs(:,ii),optprop);
            end
        end
        
        function [value,grad] = sparse_net_opt(dict,dict_transp,dict_dict_transp,instance_plain,instance_coded,S,dS,lambda,sigma)
            value = 1/2 * sum((instance_plain - dict_transp * instance_coded) .^ 2) + lambda * sum(S(instance_coded / sigma));
            grad = -dict * instance_plain + dict_dict_transp * instance_coded + lambda / sigma * dS(instance_coded / sigma);
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.dictionary".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Correlation.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'Corr',[],logg);
            
            assert(check.same(t.dict,[1 0; 0 1; 0.7071 0.7071],1e-3));
            assert(check.same(t.dict_transp,[1 0 0.7071; 0 1 0.7071],1e-3));
            assert(t.word_count == 3);
            assert(check.same(t.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.coding_params_cell,{}));
            assert(check.same(t.coding_method,'Corr'));
            assert(check.same(t.coding_params,[]));
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,3));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Correlation and Order.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'CorrOrder',[0.66 0.25 2],logg);
            
            assert(check.same(t.dict,[1 0; 0 1; 0.7071 0.7071],1e-3));
            assert(check.same(t.dict_transp,[1 0 0.7071; 0 1 0.7071],1e-3));
            assert(t.word_count == 3);
            assert(check.same(t.coding_fn,@transforms.record.dictionary.correlation_order));
            assert(check.same(t.coding_params_cell,{[1 0.25 0] 2}));
            assert(check.same(t.coding_method,'CorrOrder'));
            assert(check.same(t.coding_params,[0.66 0.25 2]));
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,3));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Matching Pursuit.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',1,logg);
            
            assert(check.same(t.dict,[1 0; 0 1; 0.7071 0.7071],1e-3));
            assert(check.same(t.dict_transp,[1 0 0.7071; 0 1 0.7071],1e-3));
            assert(t.word_count == 3);
            assert(check.same(t.coding_fn,@transforms.record.dictionary.matching_pursuit));
            assert(check.same(t.coding_params_cell,{1}));
            assert(check.same(t.coding_method,'MP'));
            assert(check.same(t.coding_params,1));
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,3));

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Orthogonal Matching Pursuit.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'OMP',1,logg);
            
            assert(check.same(t.dict,[1 0; 0 1; 0.7071 0.7071],1e-3));
            assert(check.same(t.dict_transp,[1 0 0.7071; 0 1 0.7071],1e-3));
            assert(t.word_count == 3);
            assert(check.same(t.coding_fn,@transforms.record.dictionary.ortho_matching_pursuit));
            assert(check.same(t.coding_params_cell,{1}));
            assert(check.same(t.coding_method,'OMP'));
            assert(check.same(t.coding_params,1));
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,3));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    SparseNet.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'SparseNet',0.14,logg);
            
            assert(check.same(t.dict,[1 0; 0 1; 0.7071 0.7071],1e-3));
            assert(check.same(t.dict_transp,[1 0 0.7071; 0 1 0.7071],1e-3));
            assert(t.word_count == 3);
            assert(check.same(t.coding_fn,@transforms.record.dictionary.sparse_net));
            assert(check.same(t.coding_params_cell,{0.14}));
            assert(check.same(t.coding_method,'SparseNet'));
            assert(check.same(t.coding_params,0.14));
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,3));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            fprintf('    Correlation.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'Corr',[],logg);
            s_p = t.code(s,logg);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.matrix(s_r));
            assert(check.same(size(s_r),[2 600]));
            assert(check.number(s_r));
            
            if test_figure ~= -1
                figure(test_figure);
                clf(gcf());
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                hold off;
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;

            fprintf('    Correlation and Order.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'CorrOrder',[0.66 0.25 2],logg);
            s_p = t.code(s,logg);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.matrix(s_r));
            assert(check.same(size(s_r),[2 600]));
            assert(check.number(s_r));

            if test_figure ~= -1
                figure(test_figure);
                clf(gcf());
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                hold off;
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Matching Pursuit and one kept coefficient.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',1,logg);
            s_p = t.code(s,logg);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),ones(1,600)));
            assert(check.matrix(s_r));
            assert(check.same(size(s_r),[2 600]));
            assert(check.number(s_r));
            
            if test_figure ~= -1
                figure(test_figure);
                clf(gcf());
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                hold off;
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Matching Pursuit and two kept coefficients.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',2,logg);
            s_p = t.code(s,logg);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),2*ones(1,600)));
            assert(check.matrix(s_r));
            assert(check.same(size(s_r),[2 600]));
            assert(check.number(s_r))            
            
            if test_figure ~= -1
                figure(test_figure);
                clf(gcf());
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                hold off;
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Orthogonal Matching Pursuit and one kept coefficient.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'OMP',1,logg);
            s_p = t.code(s,logg);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),ones(1,600)));
            assert(check.matrix(s_r));
            assert(check.same(size(s_r),[2 600]));
            assert(check.number(s_r));
            
            if test_figure ~= -1
                figure(test_figure);
                clf(gcf());
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                hold off;
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Orthogonal Matching Pursuit and two kept coefficients.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'OMP',2,logg);
            s_p = t.code(s,logg);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),2*ones(1,600)));
            assert(check.matrix(s_r));
            assert(check.same(size(s_r),[2 600]));
            assert(check.number(s_r));
            
            if test_figure ~= -1
                figure(test_figure);
                clf(gcf());
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                hold off;
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    SparseNet.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'SparseNet',0.14,logg);
            s_p = t.code(s,logg);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.matrix(s_r));
            assert(check.same(size(s_r),[2 600]));
            assert(check.number(s_r));
            
            if test_figure ~= -1
                figure(test_figure);
                clf(gcf());
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                hold off;
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
        end
    end
end
