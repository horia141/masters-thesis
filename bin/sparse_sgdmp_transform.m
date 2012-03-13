classdef sparse_sgdmp_transform < reversible_transform
    properties (GetAccess=public,SetAccess=immutable)
        sparse_dict;
        word_count;
        coeffs_count;
        initial_learning_rate;
        final_learning_rate;
        max_iter_count;
        input_features_count;
        output_features_count;
    end
    
    methods (Access=public)
        function [obj] = sparse_sgdmp_transform(samples,word_count,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(tc.scalar(word_count) && tc.natural(word_count) && (word_count > 0));
            assert(tc.scalar(coeffs_count) && tc.natural(coeffs_count) && ...
                    (coeffs_count > 0 && coeffs_count <= word_count));
            assert(tc.scalar(initial_learning_rate) && tc.number(initial_learning_rate) && (initial_learning_rate > 0));
            assert(tc.scalar(final_learning_rate) && tc.number(final_learning_rate) && ...
                    (final_learning_rate > 0) && (final_learning_rate < initial_learning_rate));
            assert(tc.scalar(max_iter_count) && tc.natural(max_iter_count) && (max_iter_count > 0));
            
            initial_dict = 2 * rand(samples.features_count,word_count) - 1;
            sparse_dict_t = sparse_sgdmp_transform.dict_gradient_descent(initial_dict,samples.samples,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count);
            
            obj.sparse_dict = sparse_dict_t;
            obj.word_count = word_count;
            obj.coeffs_count = coeffs_count;
            obj.initial_learning_rate = initial_learning_rate;
            obj.final_learning_rate = final_learning_rate;
            obj.max_iter_count = max_iter_count;
            obj.input_features_count = samples.features_count;
            obj.output_features_count = obj.word_count;
        end
        
        function [new_samples] = code(obj,samples)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(obj.input_features_count == samples.features_count);
            
            new_samples_t = zeros(samples.samples_count,obj.output_features_count);
            
            for i = 1:samples.samples_count
                new_samples_t(i,:) = sparse_sgdmp_transform.matching_pursuit(obj.sparse_dict,samples.samples(i,:),obj.coeffs_count)';
            end
            
            new_samples = samples_set(samples.classes,new_samples_t,samples.labels_idx);
        end
        
        function [new_samples] = decode(obj,samples)
            assert(tc.scalar(samples) && tc.samples_set(samples));
            assert(obj.output_features_count == samples.features_count);
            
            new_samples_t = zeros(samples.samples_count,obj.input_features_count);
            
            for i = 1:samples.samples_count
                new_samples_t(i,:) = obj.sparse_dict * samples.samples(i,:)';
            end
            
            new_samples = samples_set(samples.classes,new_samples_t,samples.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function [norm_dict] = normalize_dict(dict)
            norm_dict = dict;
            
            for i = 1:size(dict,2)
                norm_dict(:,i) = norm_dict(:,i) / norm(norm_dict(:,i));
            end
        end
        
        function [dict] = dict_gradient_descent(initial_dict,samples,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count)
            dict = sparse_sgdmp_transform.normalize_dict(initial_dict);
            
%             figure;
            
            for iter = 1:max_iter_count
                c_sample = samples(randi(size(samples,1)),:);
                c_coeffs = sparse_sgdmp_transform.matching_pursuit(dict,c_sample,coeffs_count);
                
                delta_dict = (c_sample' - dict * c_coeffs) * c_coeffs';
                learning_rate = initial_learning_rate * (final_learning_rate / initial_learning_rate) ^ (iter / max_iter_count);
                
                dict = dict + learning_rate * delta_dict;
                dict = sparse_sgdmp_transform.normalize_dict(dict);
                
%                 if mod(iter,10) == 0
%                     clf;
%                     hold on;
%                     scatter(samples(:,1),samples(:,2),10,'b','x');
%                     scatter(dict(1,:)',dict(2,:)',20,'r','o');
%                     scatter(c_sample(1),c_sample(2),30,'b','o');
%                     axis([-3 3 -3 3]);
%                     pause(0.1);
%                 end
            end
        end
        
        function [coeffs] = matching_pursuit(dict,sample,coeffs_count)
            coeffs = zeros(size(dict,2),1);
            sample_approx = sample;
            
            for k = 1:coeffs_count
                similarities = (sample_approx * dict) .^ 2;
                [~,best_match] = max(similarities);
                coeffs(best_match) = sample_approx * dict(:,best_match);
                sample_approx = sample_approx - coeffs(best_match) * dict(:,best_match)';
            end
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "sparse_sgdmp_transform".\n');
            
            fprintf('  Proper construction.\n');
            
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);
            
            s = samples_set({'none'},A,c);
            t = sparse_sgdmp_transform(s,4,1,0.1,0.01,1000);
            
            assert(tc.matrix(t.sparse_dict));
            assert(tc.check(size(t.sparse_dict) == [2 4]));
            assert(tc.unitreal(abs(t.sparse_dict)));
            assert(tc.check(arrayfun(@(i)utils.approx(norm(t.sparse_dict(:,i)),1),1:4)));
            assert(utils.approx(abs(t.sparse_dict(:,1)),[1;0],0.1) || ...
                   utils.approx(abs(t.sparse_dict(:,2)),[1;0],0.1) || ...
                   utils.approx(abs(t.sparse_dict(:,3)),[1;0],0.1) || ...
                   utils.approx(abs(t.sparse_dict(:,4)),[1;0],0.1));
            assert(utils.approx(abs(t.sparse_dict(:,1)),[0;1],0.1) || ...
                   utils.approx(abs(t.sparse_dict(:,2)),[0;1],0.1) || ...
                   utils.approx(abs(t.sparse_dict(:,3)),[0;1],0.1) || ...
                   utils.approx(abs(t.sparse_dict(:,4)),[0;1],0.1));
            assert(utils.approx(abs(t.sparse_dict(:,1)),[0.7071;0.7071],0.1) || ...
                   utils.approx(abs(t.sparse_dict(:,2)),[0.7071;0.7071],0.1) || ...
                   utils.approx(abs(t.sparse_dict(:,3)),[0.7071;0.7071],0.1) || ...
                   utils.approx(abs(t.sparse_dict(:,4)),[0.7071;0.7071],0.1));
            assert(t.word_count == 4);
            assert(t.coeffs_count == 1);
            assert(t.initial_learning_rate == 0.1);
            assert(t.final_learning_rate == 0.01);
            assert(t.max_iter_count == 1000);
            assert(t.input_features_count == 2);
            assert(t.output_features_count == 4);
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);
            
            s = samples_set({'none'},A,c);
            t = sparse_sgdmp_transform(s,3,1,0.1,0.01,1000);
            
            s_p = t.code(s);
            
            assert(utils.same_classes(s_p.classes,s.classes));
            assert(s_p.classes_count == 1);
            assert(tc.matrix(s_p.samples));
            assert(tc.check(size(s_p.samples) == [600 3]));
            assert(tc.number(s_p.samples));
            assert(tc.check(s_p.labels_idx == s.labels_idx));
            assert(s_p.samples_count == 600);
            assert(s_p.features_count == 3);
            
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
                title('SGDMP transformed samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);
            
            s = samples_set({'none'},A,c);
            t = sparse_sgdmp_transform(s,3,1,0.1,0.01,1000);
            
            s_p = t.code(s);
            s_r = t.decode(s_p);
            
            assert(utils.same_classes(s_r.classes,s.classes));
            assert(s_r.classes_count == 1);
            assert(tc.check(s_r.samples == s_p.samples * t.sparse_dict'));
            assert(tc.check(s_r.labels_idx == s.labels_idx));
            assert(s_r.samples_count == 600);
            assert(s_r.features_count == 2);
            
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
                title('SGDMP transformed samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'o','b');
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  More complex test (2 kept coefficients).\n');
            
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);
            
            s = samples_set({'none'},A,c);
            t = sparse_sgdmp_transform(s,3,2,0.1,0.01,1000);
            
            s_p = t.code(s);
            s_r = t.decode(s_p);
            
            assert(utils.same_classes(s_r.classes,s.classes));
            assert(s_r.classes_count == 1);
            assert(tc.check(s_r.samples == s_p.samples * t.sparse_dict'));
            assert(tc.check(s_r.labels_idx == s.labels_idx));
            assert(s_r.samples_count == 600);
            assert(s_r.features_count == 2);
            
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
                title('SGDMP transformed samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'o','b');
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
        end
    end
end
