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

    methods (Access=public)
        function [obj] = gdmp2(train_sample_plain,coding_fn,word_count,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count,logger)
            assert(tc.dataset_record(train_sample_plain));
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
            
            d = dataset.geometry(train_sample_plain);

            initial_dict = rand(d,word_count);
            [sparse_dict_t,saved_mse_t] = transforms.sparse.gdmp2.dict_gradient_descent(coding_fn,initial_dict,train_sample_plain,coeffs_count,...
                                                                                        initial_learning_rate,final_learning_rate,max_iter_count,logger);

            logger.end_node();
            
            input_geometry = d;
            output_geometry = 2 * word_count;

            obj = obj@transforms.reversible(input_geometry,output_geometry,logger);
            obj.coding_fn = coding_fn;
            obj.sparse_dict = sparse_dict_t;
            obj.word_count = word_count;
            obj.coeffs_count = coeffs_count;
            obj.initial_learning_rate = initial_learning_rate;
            obj.final_learning_rate = final_learning_rate;
            obj.max_iter_count = max_iter_count;
            obj.saved_mse = saved_mse_t;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            logger.message('Building sparse samples.');

            sample_coded_1 = obj.coding_fn(obj.sparse_dict,sample_plain,obj.coeffs_count)';
            sample_coded = [max(0,sample_coded_1) max(0,-sample_coded_1)];
        end
        
        function [sample_plain_hat] = do_decode(obj,sample_coded,logger)
            logger.message('Restoring original samples from sparse ones.');
            
            d = dataset.geometry(sample_coded);

            end_2 = d / 2;
            sample_coded_1 = sample_coded(:,1:end_2) - sample_coded(:,(end_2+1):end);
            sample_plain_hat = (obj.sparse_dict * sample_coded_1')';
        end
    end
    
    methods (Static,Access=public)
        function [coeffs] = correlation(dict,sample,coeffs_count)
            coeffs_1 = (sample * dict)';
            [~,sorted_indices] = sort(abs(coeffs_1),1,'descend');
            coeffs = zeros(size(dict,2),size(sample,1));
            
            for ii = 1:size(sample,1)
                coeffs(sorted_indices(1:coeffs_count,ii),ii) = coeffs_1(sorted_indices(1:coeffs_count,ii),ii);
            end
        end

        function [coeffs] = matching_pursuit(dict,sample,coeffs_count)
            coeffs = zeros(size(dict,2),size(sample,1));
            sample_residue = sample;
            
            for k = 1:coeffs_count
                similarities = sample_residue * dict;
                [~,best_match] = max(similarities .^ 2,[],2);
                coeffs(sub2ind(size(coeffs),best_match,(1:size(sample,1))')) = coeffs(sub2ind(size(coeffs),best_match,(1:size(sample,1))')) + similarities(sub2ind(size(similarities),(1:size(sample,1))',best_match));
                sample_residue = sample - (dict * coeffs)';
            end
        end
        
        function [coeffs] = ortho_matching_pursuit(dict,sample,coeffs_count)
            coeffs = zeros(size(dict,2),size(sample,1));
            sample_residue = sample;
            selected = zeros(size(sample,1),coeffs_count);
            
            for k = 1:coeffs_count
                similarities = sample_residue * dict;
                [~,selected(:,k)] = max(abs(similarities),[],2);
                selected_p = selected(:,1:k);
                for ii = 1:size(sample,1)
                    coeffs(selected_p(ii,:),ii) = (dict(:,selected_p(ii,:))' * dict(:,selected_p(ii,:))) \ (dict(:,selected_p(ii,:))' * sample(ii,:)');
                end
                sample_residue = sample - (dict * coeffs)';
            end
        end
    end
    
    methods (Static,Access=protected)
        function [norm_dict] = normalize_dict(dict)
            norm_dict = dict ./ repmat(sqrt(sum(dict .^ 2,1)),size(dict,1),1);
        end
        
        function [dict,saved_mse] = dict_gradient_descent(coding_fn,initial_dict,sample,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count,logger)
            logger.message('Building initial dictionary.');
            
            dict = transforms.sparse.gdmp2.normalize_dict(initial_dict);
            saved_mse = zeros(max_iter_count,1);
            sample_transp = sample';
            
            logger.beg_node('Tracking progress');
            
            for iter = 1:max_iter_count
                logger.message('Iteration %d.',iter);
                
                coeffs = coding_fn(dict,sample,coeffs_count);
                
                diff = (sample_transp - dict * coeffs);
                delta_dict = diff * coeffs';
                learning_rate = initial_learning_rate * (final_learning_rate / initial_learning_rate) ^ (iter / max_iter_count);
                
                dict = dict + learning_rate * delta_dict;
                dict = transforms.sparse.gdmp2.normalize_dict(dict);
                
                mean_error = sum(mean((sample_transp - dict * coeffs) .^ 2));
                saved_mse(iter) = mean_error;
%                 logger.message('Mean error: %.0f',mean_error);
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
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            
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
            assert(tc.checkf(@(ii)t.saved_mse(ii) <= t.saved_mse(ii-1),8:20));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,8));

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
                                                          '    Iteration 20.\n'))));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.matching_pursuit,3,1,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[600 6]));
            assert(tc.number(s_p));
            
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
                                                          'Building sparse samples.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(:,1),s(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(:,1),s_p(:,2),s_p(:,3),'o','b');
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
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.correlation,3,1,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.check(s_r == (s_p(:,1:3) - s_p(:,4:6)) * t.sparse_dict'));
            
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
                                                          'Building sparse samples.\n',...
                                                          'Restoring original samples from sparse ones.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(:,1),s(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(:,1),s_p(:,2),s_p(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r(:,1),s_r(:,2),'o','b');
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
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.correlation,3,2,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.check(s_r == (s_p(:,1:3) - s_p(:,4:6)) * t.sparse_dict'));
            
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
                                                          'Building sparse samples.\n',...
                                                          'Restoring original samples from sparse ones.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(:,1),s(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(:,1),s_p(:,2),s_p(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r(:,1),s_r(:,2),'o','b');
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
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.matching_pursuit,3,1,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.check(s_r == (s_p(:,1:3) - s_p(:,4:6)) * t.sparse_dict'));
            
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
                                                          'Building sparse samples.\n',...
                                                          'Restoring original samples from sparse ones.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(:,1),s(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(:,1),s_p(:,2),s_p(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r(:,1),s_r(:,2),'o','b');
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
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.matching_pursuit,3,2,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);

            assert(tc.check(s_r == (s_p(:,1:3) - s_p(:,4:6)) * t.sparse_dict'));
            
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
                                                          'Building sparse samples.\n',...
                                                          'Restoring original samples from sparse ones.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(:,1),s(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(:,1),s_p(:,2),s_p(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r(:,1),s_r(:,2),'o','b');
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
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.ortho_matching_pursuit,3,1,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.check(s_r == (s_p(:,1:3) - s_p(:,4:6)) * t.sparse_dict'));
            
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
                                                          'Building sparse samples.\n',...
                                                          'Restoring original samples from sparse ones.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(:,1),s(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(:,1),s_p(:,2),s_p(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r(:,1),s_r(:,2),'o','b');
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
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            
            t = transforms.sparse.gdmp2(s,@transforms.sparse.gdmp2.ortho_matching_pursuit,3,2,1e-2,1e-4,20,log);            
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);

            assert(tc.check(s_r == (s_p(:,1:3) - s_p(:,4:6)) * t.sparse_dict'));

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
                                                          'Building sparse samples.\n',...
                                                          'Restoring original samples from sparse ones.\n'))));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(:,1),s(:,2),'o','b');
                line([0;t.sparse_dict(1,1)],[0;t.sparse_dict(2,1)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,2)],[0;t.sparse_dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.sparse_dict(1,3)],[0;t.sparse_dict(2,3)],'Color','r','LineWidth',3);
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(:,1),s_p(:,2),s_p(:,3),'o','b');
                title('gdmp2 transformed samples.');
                subplot(1,3,3);
                scatter(s_r(:,1),s_r(:,2),'o','b');
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
