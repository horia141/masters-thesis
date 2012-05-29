classdef dictionary < transforms.reversible
    properties (GetAccess=public,SetAccess=immutable)
        dict;
        dict_transp;
        word_count;
        coding_fn;
        coding_params;
        coding_method;
    end

    methods (Access=public)
        function [obj] = dictionary(train_sample_plain,dict,coding_method,coding_params,logger)
            assert(tc.dataset_record(train_sample_plain));
            assert(tc.matrix(dict));
            assert(size(dict,2) == size(train_sample_plain,1)); % A BIT OF A HACK
            assert(tc.scalar(coding_method));
            assert(tc.string(coding_method));
            assert(tc.one_of(coding_method,'Corr','MP','OMP'));
            assert((tc.same(coding_method,'Corr') && tc.empty(coding_params)) || ...
                   (tc.same(coding_method,'MP') && (tc.scalar(coding_params) && ...
                                                    tc.natural(coding_params) && ...
                                                    (coding_params > 0))) || ...
                   (tc.same(coding_method,'OMP') && (tc.scalar(coding_params) && ...
                                                     tc.natural(coding_params) && ...
                                                     (coding_params > 0))));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            dict_t = transforms.record.dictionary.normalize_dict(dict);
            dict_transp_t = dict_t';
            
            word_count_t = size(dict,1);
            
            if tc.same(coding_method,'Corr')
                coding_fn_t = @transforms.record.dictionary.correlation;
                coding_params_t = {};
            elseif tc.same(coding_method,'MP')
                coding_fn_t = @transforms.record.dictionary.matching_pursuit;
                coding_params_t = {coding_params(1)};
            else
                coding_fn_t = @transforms.record.dictionary.ortho_matching_pursuit;
                coding_params_t = {coding_params(1)};
            end
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = word_count_t;
            
            obj = obj@transforms.reversible(input_geometry,output_geometry,logger);
            obj.dict = dict_t;
            obj.dict_transp = dict_transp_t;
            obj.word_count = word_count_t;
            obj.coding_fn = coding_fn_t;
            obj.coding_params = coding_params_t;
            obj.coding_method = coding_method;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            logger.message('Building sparse samples.');

            sample_coded = obj.coding_fn(obj.dict,obj.dict_transp,sample_plain,obj.coding_params{:});
        end
        
        function [sample_plain_hat] = do_decode(obj,sample_coded,logger)
            logger.message('Restoring original samples from sparse ones.');

            sample_plain_hat = obj.dict_transp * sample_coded;
        end
    end
    
    methods (Static,Access=protected)
        function [norm_dict] = normalize_dict(dict)
            norm_dict = dict ./ repmat(sqrt(sum(dict .^ 2,2)),1,size(dict,2));
        end

        function [coeffs] = correlation(dict,~,sample)
            coeffs = dict * sample;
        end
        
        function [coeffs] = matching_pursuit(dict,dict_transp,sample,coeffs_count)
            coeffs = spalloc(size(dict,1),size(sample,2),coeffs_count * size(sample,2));
            sample_residue = sample;
            
            for k = 1:coeffs_count
                similarities = dict * sample_residue;
                [~,best_match] = max(similarities .^ 2,[],1);
                coeffs = coeffs + sparse(best_match,1:size(sample,2),similarities(sub2ind(size(similarities),best_match,1:size(sample,2))),size(dict,1),size(sample,2));
                sample_residue = sample - dict_transp * coeffs;
            end
        end
        
        function [coeffs] = ortho_matching_pursuit(dict,dict_transp,sample,coeffs_count)
            coeffs = spalloc(size(dict,1),size(sample,2),coeffs_count * size(sample,2));
            sample_residue = sample;
            selected = zeros(coeffs_count,size(sample,2));
            
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
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.record.dictionary".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Correlation.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'Corr',[],log);
            
            assert(tc.same(t.dict,[1 0; 0 1; 0.7071 0.7071],'Epsilon',1e-3));
            assert(tc.same(t.dict_transp,[1 0 0.7071; 0 1 0.7071],'Epsilon',1e-3));
            assert(t.word_count == 3);
            assert(tc.same(t.coding_fn,@transforms.record.dictionary.correlation));
            assert(tc.same(t.coding_params,{}));
            assert(tc.same(t.coding_method,'Corr'));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,3));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Matching Pursuit.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',1,log);
            
            assert(tc.same(t.dict,[1 0; 0 1; 0.7071 0.7071],'Epsilon',1e-3));
            assert(tc.same(t.dict_transp,[1 0 0.7071; 0 1 0.7071],'Epsilon',1e-3));
            assert(t.word_count == 3);
            assert(tc.same(t.coding_fn,@transforms.record.dictionary.matching_pursuit));
            assert(tc.same(t.coding_params,{1}));            
            assert(tc.same(t.coding_method,'MP'));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,3));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Orthogonal Matching Pursuit.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'OMP',1,log);
            
            assert(tc.same(t.dict,[1 0; 0 1; 0.7071 0.7071],'Epsilon',1e-3));
            assert(tc.same(t.dict_transp,[1 0 0.7071; 0 1 0.7071],'Epsilon',1e-3));
            assert(t.word_count == 3);
            assert(tc.same(t.coding_fn,@transforms.record.dictionary.ortho_matching_pursuit));
            assert(tc.same(t.coding_params,{1}));
            assert(tc.same(t.coding_method,'OMP'));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,3));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    Correlation.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'Corr',[],log);
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[3 600]));
            assert(tc.number(s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Matching Pursuit and one kept coefficient.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',1,log);
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[3 600]));
            assert(tc.number(s_p));
            assert(tc.same(sum(s_p ~= 0,1),ones(1,600)));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Matching Pursuit and two kept coefficients.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',2,log);
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[3 600]));
            assert(tc.number(s_p));
            assert(tc.same(sum(s_p ~= 0,1),2*ones(1,600)));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Orthogonal Matching Pursuit and one kept coefficient.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'OMP',1,log);
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[3 600]));
            assert(tc.number(s_p));
            assert(tc.same(sum(s_p ~= 0,1),ones(1,600)));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Orthogonal Matching Pursuit and two kept coefficients.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'OMP',2,log);
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[3 600]));
            assert(tc.number(s_p));
            assert(tc.same(sum(s_p ~= 0,1),2*ones(1,600)));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            fprintf('    Correlation with three word dictionary.\n');

            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary(s,[1 0; 0 1; 0 1],'Corr',[],log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
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
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Correlation with two word dictionary.\n');

            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary(s,[1 0; 0 1],'Corr',[],log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            assert(tc.same(s,s_r));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter(s_p(1,:),s_p(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Matching Pursuit and one kept coefficient.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',1,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
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
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Matching Pursuit and two kept coefficients.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'MP',2,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
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
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Orthogonal Matching Pursuit and one kept coefficient.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'OMP',1,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
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
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Orthogonal Matching Pursuit and two kept coefficients.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary(s,[1 0; 0 1; 1 1],'OMP',2,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
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
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
