classdef dictionary < transform
    properties (GetAccess=public,SetAccess=immutable)
        dict;
        dict_transp;
        dict_x_dict_transp;
        word_count;
        coding_fn;
        coding_params_cell;
        coding_method;
        coding_params;
        coeff_count;
        num_workers;
    end

    methods (Access=public)
        function [obj] = dictionary(train_sample_plain,dict,coding_method,coding_params,coeff_count,num_workers)
            assert(check.dataset_record(train_sample_plain));
            assert(check.matrix(dict));
            assert(size(dict,2) == dataset.geometry(train_sample_plain));
            assert(transforms.record.dictionary.coding_setup_ok(coding_method,coding_params));
            assert(check.scalar(coeff_count));
            assert(check.natural(coeff_count));
            assert(coeff_count >= 1);
            assert(coeff_count <= size(dict,1));
            assert(check.scalar(num_workers));
            assert(check.natural(num_workers));
            assert(num_workers >= 1);
            
            dict_t = transforms.record.dictionary.normalize_dict(dict);
            dict_transp_t = dict_t';
            dict_x_dict_transp_t = dict_t * dict_transp_t;
            word_count_t = size(dict,1);
            [coding_fn_t,coding_params_cell_t] = transforms.record.dictionary.coding_setup(coding_method);
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = word_count_t;

            obj = obj@transform(input_geometry,output_geometry);
            obj.dict = dict_t;
            obj.dict_transp = dict_transp_t;
            obj.dict_x_dict_transp = dict_x_dict_transp_t;
            obj.word_count = word_count_t;
            obj.coding_fn = coding_fn_t;
            obj.coding_params_cell = coding_params_cell_t;
            obj.coding_method = coding_method;
            obj.coding_params = coding_params;            
            obj.coeff_count = coeff_count;
            obj.num_workers = num_workers;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,~)
            sample_coded = obj.coding_fn(obj.dict,obj.dict_transp,obj.dict_x_dict_transp,obj.coding_params_cell{:},obj.coeff_count,sample_plain,obj.num_workers);
        end
    end
    
    methods (Static,Access=protected)
        function [o] = coding_setup_ok(coding_method,coding_params)
            o = true;
            o = o && check.scalar(coding_method);
            o = o && check.string(coding_method);
            o = o && check.one_of(coding_method,'Corr','MP');
            o = o && ((check.same(coding_method,'Corr') && check.empty(coding_params)) || ...
                      (check.same(coding_method,'MP') && (check.empty(coding_params))));
        end

        function [coding_fn,coding_params_cell] = coding_setup(coding_method)
            if check.same(coding_method,'Corr')
                coding_fn = @xtern.x_dictionary_correlation;
                coding_params_cell = {[]};
            elseif check.same(coding_method,'MP')
                coding_fn = @xtern.x_dictionary_matching_pursuit;
                coding_params_cell = {[]};
            else
                assert(false);
            end
        end

        function [norm_dict] = normalize_dict(dict)
            norm_dict = dict ./ repmat(sqrt(sum(dict .^ 2,2)),1,size(dict,2));
        end
    end
    
    methods (Static,Access=public)
        function [coeffs] = reference_correlation(dict,coeff_count,observation)
            coeffs_t1 = dict * observation;
            [~,coeffs_idx_t1] = sort(abs(coeffs_t1),'descend');
            coeffs_idx_t2 = coeffs_idx_t1(1:coeff_count);
            coeffs = zeros(size(dict,1),1);
            coeffs(coeffs_idx_t2) = coeffs_t1(coeffs_idx_t2);
        end
        
        function [coeffs] = reference_correlation_order(dict,modulator,coeff_count,observation)
            coeffs_t1 = dict * observation;
            [~,coeffs_idx_t1] = sort(abs(coeffs_t1),'descend');
            coeffs_idx_t2 = coeffs_idx_t1(1:coeff_count);
            coeffs = zeros(size(dict,1),1);
            coeffs(coeffs_idx_t2) = sign(coeffs_t1(coeffs_idx_t2)) .* modulator';
        end
        
        function [coeffs] = reference_matching_pursuit(dict,coeff_count,observation)
            residual = observation;
            coeffs = zeros(size(dict,1),1);
            
            for k = 1:coeff_count
                similarities = dict * residual;
                [~,omega_similarity] = max(abs(similarities));
                coeffs(omega_similarity) = similarities(omega_similarity);
                residual = observation - dict' * coeffs;
            end            
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.dictionary".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Correlation.\n');
            
            s = dataset.load('../test/three_component_cloud.mat');
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'Corr',[],3,1);
            
            assert(check.same(t.dict,[1 0; 0 1; 0.7071 0.7071],1e-3));
            assert(check.same(t.dict_transp,[1 0 0.7071; 0 1 0.7071],1e-3));
            assert(check.same(t.dict_x_dict_transp,[1 0; 0 1; 0.7071 0.7071] * [1 0 0.7071; 0 1 0.7071],1e-3));
            assert(t.word_count == 3);
            assert(check.same(t.coding_fn,@xtern.x_dictionary_correlation));
            assert(check.same(t.coding_params_cell,{[]}));
            assert(check.same(t.coding_method,'Corr'));
            assert(check.same(t.coding_params,[]));
            assert(t.coeff_count == 3);
            assert(t.num_workers == 1);
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,3));
            
            clearvars -except test_figure;
            
            fprintf('    Matching Pursuit.\n');
            
            s = dataset.load('../test/three_component_cloud.mat');
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',[],3,1);
            
            assert(check.same(t.dict,[1 0; 0 1; 0.7071 0.7071],1e-3));
            assert(check.same(t.dict_transp,[1 0 0.7071; 0 1 0.7071],1e-3));
            assert(check.same(t.dict_x_dict_transp,[1 0; 0 1; 0.7071 0.7071] * [1 0 0.7071; 0 1 0.7071],1e-3));
            assert(t.word_count == 3);
            assert(check.same(t.coding_fn,@xtern.x_dictionary_matching_pursuit));
            assert(check.same(t.coding_params_cell,{[]}));
            assert(check.same(t.coding_method,'MP'));
            assert(check.same(t.coding_params,[]));
            assert(t.coeff_count == 3);
            assert(t.num_workers == 1);
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,3));

            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            fprintf('    Correlation with one kept coefficient and single thread.\n');
            
            dict_t = transforms.record.dictionary.normalize_dict([1 0; 0 1; 1 1]);
            s = dataset.load('../test/three_component_cloud.mat');
            
            t = transforms.record.dictionary(s,dict_t,'Corr',[],1,1);
            s_p = t.code(s);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),ones(1,600)));
            assert(check.same(s_p,sparse(cell2mat(arrayfun(@(ii)transforms.record.dictionary.reference_correlation(dict_t,1,s(:,ii)),1:600,'UniformOutput',false))),1e-6));
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
            
            clearvars -except test_figure;
            
            fprintf('    Correlation with one kept coefficient and multiple threads.\n');
            
            dict_t = transforms.record.dictionary.normalize_dict([1 0; 0 1; 1 1]);
            s = dataset.load('../test/three_component_cloud.mat');
            
            t = transforms.record.dictionary(s,dict_t,'Corr',[],1,10);
            s_p = t.code(s);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),ones(1,600)));
            assert(check.same(s_p,sparse(cell2mat(arrayfun(@(ii)transforms.record.dictionary.reference_correlation(dict_t,1,s(:,ii)),1:600,'UniformOutput',false))),1e-6));
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
            
            clearvars -except test_figure;
            
            fprintf('    Correlation with two kept coefficients and single thread.\n');
            
            dict_t = transforms.record.dictionary.normalize_dict([1 0; 0 1; 1 1]);
            s = dataset.load('../test/three_component_cloud.mat');
            
            t = transforms.record.dictionary(s,dict_t,'Corr',[],2,1);
            s_p = t.code(s);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),2*ones(1,600)));
            assert(check.same(s_p,sparse(cell2mat(arrayfun(@(ii)transforms.record.dictionary.reference_correlation(dict_t,2,s(:,ii)),1:600,'UniformOutput',false))),1e-6));
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
            
            clearvars -except test_figure;
            
            fprintf('    Correlation with two kept coefficients and multiple threads.\n');
            
            dict_t = transforms.record.dictionary.normalize_dict([1 0; 0 1; 1 1]);
            s = dataset.load('../test/three_component_cloud.mat');
            
            t = transforms.record.dictionary(s,dict_t,'Corr',[],2,10);
            s_p = t.code(s);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),2*ones(1,600)));
            assert(check.same(s_p,sparse(cell2mat(arrayfun(@(ii)transforms.record.dictionary.reference_correlation(dict_t,2,s(:,ii)),1:600,'UniformOutput',false))),1e-6));
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
            
            clearvars -except test_figure;
            
            fprintf('    Matching Pursuit with one kept coefficient and single thread.\n');
            
            dict_t = transforms.record.dictionary.normalize_dict([1 0; 0 1; 1 1]);
            s = dataset.load('../test/three_component_cloud.mat');
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',[],1,1);
            s_p = t.code(s);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),ones(1,600)));
            assert(check.same(s_p,sparse(cell2mat(arrayfun(@(ii)transforms.record.dictionary.reference_matching_pursuit(dict_t,1,s(:,ii)),1:600,'UniformOutput',false))),1e-6));
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
            
            clearvars -except test_figure;
            
            fprintf('    Matching Pursuit with one kept coefficient and multiple threads.\n');
            
            dict_t = transforms.record.dictionary.normalize_dict([1 0; 0 1; 1 1]);
            s = dataset.load('../test/three_component_cloud.mat');
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',[],1,10);
            s_p = t.code(s);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),ones(1,600)));
            assert(check.same(s_p,sparse(cell2mat(arrayfun(@(ii)transforms.record.dictionary.reference_matching_pursuit(dict_t,1,s(:,ii)),1:600,'UniformOutput',false))),1e-6));
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
            
            clearvars -except test_figure;
            
            fprintf('    Matching Pursuit with two kept coefficients and single thread.\n');
            
            dict_t = transforms.record.dictionary.normalize_dict([1 0; 0 1; 1 1]);
            s = dataset.load('../test/three_component_cloud.mat');
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',[],2,1);
            s_p = t.code(s);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),2*ones(1,600)));
            assert(check.same(s_p,sparse(cell2mat(arrayfun(@(ii)transforms.record.dictionary.reference_matching_pursuit(dict_t,2,s(:,ii)),1:600,'UniformOutput',false))),1e-6));
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
            
            clearvars -except test_figure;
            
            fprintf('    Matching Pursuit with two kept coefficients and multiple threads.\n');
            
            dict_t = transforms.record.dictionary.normalize_dict([1 0; 0 1; 1 1]);
            s = dataset.load('../test/three_component_cloud.mat');
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',[],2,10);
            s_p = t.code(s);
            s_r = t.dict_transp * s_p;
            
            assert(check.matrix(s_p));
            assert(check.same(size(s_p),[3 600]));
            assert(check.number(s_p));
            assert(check.same(sum(s_p ~= 0,1),2*ones(1,600)));
            assert(check.same(s_p,sparse(cell2mat(arrayfun(@(ii)transforms.record.dictionary.reference_matching_pursuit(dict_t,2,s(:,ii)),1:600,'UniformOutput',false))),1e-6));
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
            
            clearvars -except test_figure;
        end
    end
end
