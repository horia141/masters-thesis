classdef gdmp2 < transforms.reversible
    properties (GetAccess=public,SetAccess=immutable)
        coding_fn;
        sparse_dict;
        word_count;
        coeffs_count;
        initial_learning_rate;
        final_learning_rate;
        max_iter_count;
        saved_mse;
    end
    
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = gdmp2(train_dataset_plain,coding_fn,word_count,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count,logger)
            assert(tc.scalar(train_dataset_plain));
            assert(tc.dataset(train_dataset_plain));
            assert(train_dataset_plain.samples_count >= 1);
            assert(tc.scalar(coding_fn));
            assert(tc.function_h(coding_fn));
            assert(tc.one_of(coding_fn,@transforms.sparse.gdmp2.correlation,@transforms.sparse.gdmp2.matching_pursuit,@transforms.sparse.gdmp2.ortho_matching_pursuit));
            assert(tc.scalar(word_count));
            assert(tc.natural(word_count));
            assert(word_count >= 1);
            assert(tc.scalar(coeffs_count));
            assert(tc.natural(coeffs_count));
            assert(coeffs_count >= 1 && coeffs_count <= word_count);
            assert(tc.scalar(initial_learning_rate));
            assert(tc.number(initial_learning_rate));
            assert(initial_learning_rate > 0);
            assert(tc.scalar(final_learning_rate));
            assert(tc.number(final_learning_rate));
            assert(final_learning_rate > 0);
            assert(final_learning_rate <= initial_learning_rate);
            assert(tc.scalar(max_iter_count));
            assert(tc.natural(max_iter_count));
            assert(max_iter_count >= 1);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            logger.beg_node('Learning sparse dictionary');
            
            initial_dict = rand(train_dataset_plain.features_count,word_count);
            [sparse_dict_t,saved_mse_t] = transforms.sparse.gdmp2.dict_gradient_descent(coding_fn,initial_dict,train_dataset_plain.samples,coeffs_count,...
                                                                                     initial_learning_rate,final_learning_rate,max_iter_count,logger);
                                                                     
            logger.end_node();
                                                                     
            obj = obj@transforms.reversible(logger);
            obj.coding_fn = coding_fn;
            obj.sparse_dict = sparse_dict_t;
            obj.word_count = word_count;
            obj.coeffs_count = coeffs_count;
            obj.initial_learning_rate = initial_learning_rate;
            obj.final_learning_rate = final_learning_rate;
            obj.max_iter_count = max_iter_count;
            obj.saved_mse = saved_mse_t;
            
            logger.beg_node('Extracting plain/coded samples');
            
            obj.one_sample_plain = train_dataset_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain,logger);
            
            logger.end_node();
        end
    end
    
    methods (Access=protected)
        function [dataset_coded] = do_code(obj,dataset_plain,logger)
            logger.message('Building sparse samples.');

            samples_coded_1 = obj.coding_fn(obj.sparse_dict,dataset_plain.samples,obj.coeffs_count)';
            samples_coded = [max(0,samples_coded_1) max(0,-samples_coded_1)];

            logger.message('Building dataset.');

            dataset_coded = dataset(dataset_plain.classes,samples_coded,dataset_plain.labels_idx);
        end
        
        function [dataset_plain_hat] = do_decode(obj,dataset_coded,logger)
            logger.message('Restoring original samples from sparse ones.');

            end_2 = size(dataset_coded.samples,2) / 2;
            samples_coded_1 = dataset_coded.samples(:,1:end_2) - dataset_coded.samples(:,(end_2+1):end);
            samples_plain_hat = (obj.sparse_dict * samples_coded_1')';

            logger.message('Building dataset.');
            
            dataset_plain_hat = dataset(dataset_coded.classes,samples_plain_hat,dataset_coded.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function [coeffs] = correlation(dict,samples,coeffs_count)
            coeffs_1 = (samples * dict)';
            [~,sorted_indices] = sort(abs(coeffs_1),1,'descend');
            coeffs = zeros(size(dict,2),size(samples,1));
            
            for ii = 1:size(samples,1)
                coeffs(sorted_indices(1:coeffs_count,ii),ii) = coeffs_1(sorted_indices(1:coeffs_count,ii),ii);
            end
        end

        function [coeffs] = matching_pursuit(dict,samples,coeffs_count)
            coeffs = zeros(size(dict,2),size(samples,1));
            samples_residue = samples;
            
            for k = 1:coeffs_count
                similarities = samples_residue * dict;
                [~,best_match] = max(similarities .^ 2,[],2);
                coeffs(sub2ind(size(coeffs),best_match,(1:size(samples,1))')) = coeffs(sub2ind(size(coeffs),best_match,(1:size(samples,1))')) + similarities(sub2ind(size(similarities),(1:size(samples,1))',best_match));
                samples_residue = samples - (dict * coeffs)';
            end
        end
        
        function [coeffs] = ortho_matching_pursuit(dict,samples,coeffs_count)
            coeffs = zeros(size(dict,2),size(samples,1));
            samples_residue = samples;
            selected = zeros(size(samples,1),coeffs_count);
            
            for k = 1:coeffs_count
                similarities = samples_residue * dict;
                [~,selected(:,k)] = max(abs(similarities),[],2);
                selected_p = selected(:,1:k);
                for ii = 1:size(samples,1)
                    coeffs(selected_p(ii,:),ii) = (dict(:,selected_p(ii,:))' * dict(:,selected_p(ii,:))) \ (dict(:,selected_p(ii,:))' * samples(ii,:)');
                end
                samples_residue = samples - (dict * coeffs)';
            end
        end
    end
    
    methods (Static,Access=protected)
        function [norm_dict] = normalize_dict(dict)
            norm_dict = dict ./ repmat(sqrt(sum(dict .^ 2,1)),size(dict,1),1);
        end
        
        function [dict,saved_mse] = dict_gradient_descent(coding_fn,initial_dict,samples,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count,logger)
            logger.message('Building initial dictionary.');
            
            dict = transforms.sparse.gdmp2.normalize_dict(initial_dict);
            saved_mse = zeros(max_iter_count,1);
            samples_transp = samples';
            
            logger.beg_node('Tracking progress');
            
            for iter = 1:max_iter_count
                logger.message('Iteration %d.',iter);
                
                coeffs = coding_fn(dict,samples,coeffs_count);
                
                diff = (samples_transp - dict * coeffs);
                delta_dict = diff * coeffs';
                learning_rate = initial_learning_rate * (final_learning_rate / initial_learning_rate) ^ (iter / max_iter_count);
                
                dict = dict + learning_rate * delta_dict;
                dict = transforms.sparse.gdmp2.normalize_dict(dict);
                
                mean_error = sum(mean((samples_transp - dict * coeffs) .^ 2));
                saved_mse(iter) = mean_error;
                logger.message('Mean error: %.0f',mean_error);
%                 sz = sqrt(size(initial_dict,1));
%                 im_dict = zeros(sz,sz,1,size(initial_dict,2));
%                 for ii = 1:size(initial_dict,2)
%                    im_dict(:,:,1,ii) = reshape(dict(:,ii),sz,sz);
%                 end
%                 imshow(utils.format_as_tiles(utils.remap_images_to_unit(im_dict,'global')));
%                 pause;
            end
            
            logger.end_node();
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.sparse.gdmp2".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.matching_pursuit,4,1,1e-2,1e-4,20,log);
            
            assert(tc.same(t.coding_fn,@transforms.sparse.gdmp2.matching_pursuit));
            assert(tc.check(size(t.sparse_dict) == [2 4]));
            assert(tc.matrix(t.sparse_dict) && tc.unitreal(abs(t.sparse_dict)));
            assert(tc.check(arrayfun(@(ii)tc.same(norm(t.sparse_dict(:,ii)),1),1:4)));
            assert(tc.same(abs(t.sparse_dict(:,1)),[1;0],'Epsilon',0.1) || ...
                   tc.same(abs(t.sparse_dict(:,2)),[1;0],'Epsilon',0.1) || ...
                   tc.same(abs(t.sparse_dict(:,3)),[1;0],'Epsilon',0.1) || ...
                   tc.same(abs(t.sparse_dict(:,4)),[1;0],'Epsilon',0.1));
            assert(tc.same(abs(t.sparse_dict(:,1)),[0;1],'Epsilon',0.1) || ...
                   tc.same(abs(t.sparse_dict(:,2)),[0;1],'Epsilon',0.1) || ...
                   tc.same(abs(t.sparse_dict(:,3)),[0;1],'Epsilon',0.1) || ...
                   tc.same(abs(t.sparse_dict(:,4)),[0;1],'Epsilon',0.1));
            assert(tc.same(abs(t.sparse_dict(:,1)),[0.7071;0.7071],'Epsilon',0.1) || ...
                   tc.same(abs(t.sparse_dict(:,2)),[0.7071;0.7071],'Epsilon',0.1) || ...
                   tc.same(abs(t.sparse_dict(:,3)),[0.7071;0.7071],'Epsilon',0.1) || ...
                   tc.same(abs(t.sparse_dict(:,4)),[0.7071;0.7071],'Epsilon',0.1));
            assert(t.word_count == 4);
            assert(t.coeffs_count == 1);
            assert(t.initial_learning_rate == 1e-2);
            assert(t.final_learning_rate == 1e-4);
            assert(t.max_iter_count == 20);
            assert(tc.vector(t.saved_mse));
            assert(length(t.saved_mse) == 20);
            assert(tc.number(t.saved_mse));
            assert(tc.check(t.saved_mse > 0));
            assert(tc.checkf(@(ii)t.saved_mse(ii) <= t.saved_mse(ii-1),5:20));
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
            assert(tc.check(size(t.one_sample_coded.samples) == [1 8]));
            assert(tc.matrix(t.one_sample_coded.samples) && tc.number(t.one_sample_coded.samples));
            assert(tc.check(t.one_sample_coded.labels_idx == c(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 8);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Learning sparse dictionary:\n',...
                                                          '  Building initial dictionary.\n',...
                                                          '  Tracking progress:\n',...
                                                          '    Iteration 1.\n',...
                                                          '    Iteration 2.\n',...
                                                          '    Iteration 3.\n',...
                                                          '    Iteration 4.\n',...
                                                          '    Iteration 5.\n',...
                                                          '    Iteration 6.\n',...
                                                          '    Iteration 7.\n',...
                                                          '    Iteration 8.\n',...
                                                          '    Iteration 9.\n',...
                                                          '    Iteration 10.\n',...
                                                          '    Iteration 11.\n',...
                                                          '    Iteration 12.\n',...
                                                          '    Iteration 13.\n',...
                                                          '    Iteration 14.\n',...
                                                          '    Iteration 15.\n',...
                                                          '    Iteration 16.\n',...
                                                          '    Iteration 17.\n',...
                                                          '    Iteration 18.\n',...
                                                          '    Iteration 19.\n',...
                                                          '    Iteration 20.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Building sparse samples.\n',...
                                                          '  Building dataset.\n'))));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.matching_pursuit,3,1,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            
            assert(tc.same(s_p.classes,s.classes));
            assert(s_p.classes_count == 1);
            assert(tc.matrix(s_p.samples));
            assert(tc.check(size(s_p.samples) == [600 6]));
            assert(tc.number(s_p.samples));
            assert(tc.check(s_p.labels_idx == s.labels_idx));
            assert(s_p.samples_count == 600);
            assert(s_p.features_count == 6);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Learning sparse dictionary:\n',...
                                                          '  Building initial dictionary.\n',...
                                                          '  Tracking progress:\n',...
                                                          '    Iteration 1.\n',...
                                                          '    Iteration 2.\n',...
                                                          '    Iteration 3.\n',...
                                                          '    Iteration 4.\n',...
                                                          '    Iteration 5.\n',...
                                                          '    Iteration 6.\n',...
                                                          '    Iteration 7.\n',...
                                                          '    Iteration 8.\n',...
                                                          '    Iteration 9.\n',...
                                                          '    Iteration 10.\n',...
                                                          '    Iteration 11.\n',...
                                                          '    Iteration 12.\n',...
                                                          '    Iteration 13.\n',...
                                                          '    Iteration 14.\n',...
                                                          '    Iteration 15.\n',...
                                                          '    Iteration 16.\n',...
                                                          '    Iteration 17.\n',...
                                                          '    Iteration 18.\n',...
                                                          '    Iteration 19.\n',...
                                                          '    Iteration 20.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Building sparse samples.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building sparse samples.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s.samples(:,1),s.samples(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p.samples(:,1),s_p.samples(:,2),s_p.samples(:,3),'o','b');
                title('gdmp2 transformed samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            fprintf('    With one kept coefficient and Correlation.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.correlation,3,1,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.same(s_r.classes,s.classes));
            assert(s_r.classes_count == 1);
            assert(tc.check(s_r.samples == (s_p.samples(:,1:3) - s_p.samples(:,4:6)) * t.sparse_dict'));
            assert(tc.check(s_r.labels_idx == s.labels_idx));
            assert(s_r.samples_count == 600);
            assert(s_r.features_count == 2);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Learning sparse dictionary:\n',...
                                                          '  Building initial dictionary.\n',...
                                                          '  Tracking progress:\n',...
                                                          '    Iteration 1.\n',...
                                                          '    Iteration 2.\n',...
                                                          '    Iteration 3.\n',...
                                                          '    Iteration 4.\n',...
                                                          '    Iteration 5.\n',...
                                                          '    Iteration 6.\n',...
                                                          '    Iteration 7.\n',...
                                                          '    Iteration 8.\n',...
                                                          '    Iteration 9.\n',...
                                                          '    Iteration 10.\n',...
                                                          '    Iteration 11.\n',...
                                                          '    Iteration 12.\n',...
                                                          '    Iteration 13.\n',...
                                                          '    Iteration 14.\n',...
                                                          '    Iteration 15.\n',...
                                                          '    Iteration 16.\n',...
                                                          '    Iteration 17.\n',...
                                                          '    Iteration 18.\n',...
                                                          '    Iteration 19.\n',...
                                                          '    Iteration 20.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Building sparse samples.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building sparse samples.\n',...
                                                          'Building dataset.\n',...
                                                          'Restoring original samples from sparse ones.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s.samples(:,1),s.samples(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p.samples(:,1),s_p.samples(:,2),s_p.samples(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'o','b');
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With 2 kept coefficients and Correlation.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.correlation,3,2,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.same(s_r.classes,s.classes));
            assert(s_r.classes_count == 1);
            assert(tc.check(s_r.samples == (s_p.samples(:,1:3) - s_p.samples(:,4:6)) * t.sparse_dict'));
            assert(tc.check(s_r.labels_idx == s.labels_idx));
            assert(s_r.samples_count == 600);
            assert(s_r.features_count == 2);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Learning sparse dictionary:\n',...
                                                          '  Building initial dictionary.\n',...
                                                          '  Tracking progress:\n',...
                                                          '    Iteration 1.\n',...
                                                          '    Iteration 2.\n',...
                                                          '    Iteration 3.\n',...
                                                          '    Iteration 4.\n',...
                                                          '    Iteration 5.\n',...
                                                          '    Iteration 6.\n',...
                                                          '    Iteration 7.\n',...
                                                          '    Iteration 8.\n',...
                                                          '    Iteration 9.\n',...
                                                          '    Iteration 10.\n',...
                                                          '    Iteration 11.\n',...
                                                          '    Iteration 12.\n',...
                                                          '    Iteration 13.\n',...
                                                          '    Iteration 14.\n',...
                                                          '    Iteration 15.\n',...
                                                          '    Iteration 16.\n',...
                                                          '    Iteration 17.\n',...
                                                          '    Iteration 18.\n',...
                                                          '    Iteration 19.\n',...
                                                          '    Iteration 20.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Building sparse samples.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building sparse samples.\n',...
                                                          'Building dataset.\n',...
                                                          'Restoring original samples from sparse ones.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s.samples(:,1),s.samples(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p.samples(:,1),s_p.samples(:,2),s_p.samples(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'o','b');
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With one kept coefficient and MP.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.matching_pursuit,3,1,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.same(s_r.classes,s.classes));
            assert(s_r.classes_count == 1);
            assert(tc.check(s_r.samples == (s_p.samples(:,1:3) - s_p.samples(:,4:6)) * t.sparse_dict'));
            assert(tc.check(s_r.labels_idx == s.labels_idx));
            assert(s_r.samples_count == 600);
            assert(s_r.features_count == 2);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Learning sparse dictionary:\n',...
                                                          '  Building initial dictionary.\n',...
                                                          '  Tracking progress:\n',...
                                                          '    Iteration 1.\n',...
                                                          '    Iteration 2.\n',...
                                                          '    Iteration 3.\n',...
                                                          '    Iteration 4.\n',...
                                                          '    Iteration 5.\n',...
                                                          '    Iteration 6.\n',...
                                                          '    Iteration 7.\n',...
                                                          '    Iteration 8.\n',...
                                                          '    Iteration 9.\n',...
                                                          '    Iteration 10.\n',...
                                                          '    Iteration 11.\n',...
                                                          '    Iteration 12.\n',...
                                                          '    Iteration 13.\n',...
                                                          '    Iteration 14.\n',...
                                                          '    Iteration 15.\n',...
                                                          '    Iteration 16.\n',...
                                                          '    Iteration 17.\n',...
                                                          '    Iteration 18.\n',...
                                                          '    Iteration 19.\n',...
                                                          '    Iteration 20.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Building sparse samples.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building sparse samples.\n',...
                                                          'Building dataset.\n',...
                                                          'Restoring original samples from sparse ones.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s.samples(:,1),s.samples(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p.samples(:,1),s_p.samples(:,2),s_p.samples(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'o','b');
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With 2 kept coefficients and MP.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.matching_pursuit,3,2,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.same(s_r.classes,s.classes));
            assert(s_r.classes_count == 1);
            assert(tc.check(s_r.samples == (s_p.samples(:,1:3) - s_p.samples(:,4:6)) * t.sparse_dict'));
            assert(tc.check(s_r.labels_idx == s.labels_idx));
            assert(s_r.samples_count == 600);
            assert(s_r.features_count == 2);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Learning sparse dictionary:\n',...
                                                          '  Building initial dictionary.\n',...
                                                          '  Tracking progress:\n',...
                                                          '    Iteration 1.\n',...
                                                          '    Iteration 2.\n',...
                                                          '    Iteration 3.\n',...
                                                          '    Iteration 4.\n',...
                                                          '    Iteration 5.\n',...
                                                          '    Iteration 6.\n',...
                                                          '    Iteration 7.\n',...
                                                          '    Iteration 8.\n',...
                                                          '    Iteration 9.\n',...
                                                          '    Iteration 10.\n',...
                                                          '    Iteration 11.\n',...
                                                          '    Iteration 12.\n',...
                                                          '    Iteration 13.\n',...
                                                          '    Iteration 14.\n',...
                                                          '    Iteration 15.\n',...
                                                          '    Iteration 16.\n',...
                                                          '    Iteration 17.\n',...
                                                          '    Iteration 18.\n',...
                                                          '    Iteration 19.\n',...
                                                          '    Iteration 20.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Building sparse samples.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building sparse samples.\n',...
                                                          'Building dataset.\n',...
                                                          'Restoring original samples from sparse ones.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s.samples(:,1),s.samples(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p.samples(:,1),s_p.samples(:,2),s_p.samples(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'o','b');
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With one kept coefficient and OMP.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.ortho_matching_pursuit,3,1,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.same(s_r.classes,s.classes));
            assert(s_r.classes_count == 1);
            assert(tc.check(s_r.samples == (s_p.samples(:,1:3) - s_p.samples(:,4:6)) * t.sparse_dict'));
            assert(tc.check(s_r.labels_idx == s.labels_idx));
            assert(s_r.samples_count == 600);
            assert(s_r.features_count == 2);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Learning sparse dictionary:\n',...
                                                          '  Building initial dictionary.\n',...
                                                          '  Tracking progress:\n',...
                                                          '    Iteration 1.\n',...
                                                          '    Iteration 2.\n',...
                                                          '    Iteration 3.\n',...
                                                          '    Iteration 4.\n',...
                                                          '    Iteration 5.\n',...
                                                          '    Iteration 6.\n',...
                                                          '    Iteration 7.\n',...
                                                          '    Iteration 8.\n',...
                                                          '    Iteration 9.\n',...
                                                          '    Iteration 10.\n',...
                                                          '    Iteration 11.\n',...
                                                          '    Iteration 12.\n',...
                                                          '    Iteration 13.\n',...
                                                          '    Iteration 14.\n',...
                                                          '    Iteration 15.\n',...
                                                          '    Iteration 16.\n',...
                                                          '    Iteration 17.\n',...
                                                          '    Iteration 18.\n',...
                                                          '    Iteration 19.\n',...
                                                          '    Iteration 20.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Building sparse samples.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building sparse samples.\n',...
                                                          'Building dataset.\n',...
                                                          'Restoring original samples from sparse ones.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s.samples(:,1),s.samples(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p.samples(:,1),s_p.samples(:,2),s_p.samples(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'o','b');
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With 2 kept coefficients and OMP.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.ortho_matching_pursuit,3,2,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.same(s_r.classes,s.classes));
            assert(s_r.classes_count == 1);
            assert(tc.check(s_r.samples == (s_p.samples(:,1:3) - s_p.samples(:,4:6)) * t.sparse_dict'));
            assert(tc.check(s_r.labels_idx == s.labels_idx));
            assert(s_r.samples_count == 600);
            assert(s_r.features_count == 2);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Learning sparse dictionary:\n',...
                                                          '  Building initial dictionary.\n',...
                                                          '  Tracking progress:\n',...
                                                          '    Iteration 1.\n',...
                                                          '    Iteration 2.\n',...
                                                          '    Iteration 3.\n',...
                                                          '    Iteration 4.\n',...
                                                          '    Iteration 5.\n',...
                                                          '    Iteration 6.\n',...
                                                          '    Iteration 7.\n',...
                                                          '    Iteration 8.\n',...
                                                          '    Iteration 9.\n',...
                                                          '    Iteration 10.\n',...
                                                          '    Iteration 11.\n',...
                                                          '    Iteration 12.\n',...
                                                          '    Iteration 13.\n',...
                                                          '    Iteration 14.\n',...
                                                          '    Iteration 15.\n',...
                                                          '    Iteration 16.\n',...
                                                          '    Iteration 17.\n',...
                                                          '    Iteration 18.\n',...
                                                          '    Iteration 19.\n',...
                                                          '    Iteration 20.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Building sparse samples.\n',...
                                                          '  Building dataset.\n',...
                                                          'Building sparse samples.\n',...
                                                          'Building dataset.\n',...
                                                          'Restoring original samples from sparse ones.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s.samples(:,1),s.samples(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p.samples(:,1),s_p.samples(:,2),s_p.samples(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'o','b');
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
