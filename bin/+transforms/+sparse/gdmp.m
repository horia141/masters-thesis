classdef gdmp < transforms.reversible
    properties (GetAccess=public,SetAccess=immutable)
        sparse_dict;
        word_count;
        coeffs_count;
        initial_learning_rate;
        final_learning_rate;
        max_iter_count;
    end
    
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = gdmp(train_dataset_plain,word_count,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count)
            assert(tc.scalar(train_dataset_plain) && tc.dataset(train_dataset_plain));
            assert(train_dataset_plain.samples_count >= 1);
            assert(tc.scalar(word_count) && tc.natural(word_count) && (word_count > 0));
            assert(tc.scalar(coeffs_count) && tc.natural(coeffs_count) && ...
                    (coeffs_count > 0 && coeffs_count <= word_count));
            assert(tc.scalar(initial_learning_rate) && tc.number(initial_learning_rate) && (initial_learning_rate > 0));
            assert(tc.scalar(final_learning_rate) && tc.number(final_learning_rate) && ...
                    (final_learning_rate > 0) && (final_learning_rate < initial_learning_rate));
            assert(tc.scalar(max_iter_count) && tc.natural(max_iter_count) && (max_iter_count > 0));
            
            initial_dict = rand(train_dataset_plain.features_count,word_count);
            sparse_dict_t = transforms.sparse.gdmp.dict_gradient_descent(initial_dict,train_dataset_plain.samples,coeffs_count,...
                                                                         initial_learning_rate,final_learning_rate,max_iter_count);
                                                                     
            obj = obj@transforms.reversible();
            obj.sparse_dict = sparse_dict_t;
            obj.word_count = word_count;
            obj.coeffs_count = coeffs_count;
            obj.initial_learning_rate = initial_learning_rate;
            obj.final_learning_rate = final_learning_rate;
            obj.max_iter_count = max_iter_count;
            obj.one_sample_plain = train_dataset_plain.subsamples(1);
            obj.one_sample_coded = dataset(obj.one_sample_plain.classes,...
                                           transforms.sparse.gdmp.matching_pursuit(obj.sparse_dict,obj.one_sample_plain.samples,obj.coeffs_count)',...
                                           obj.one_sample_plain.labels_idx);
        end
    end
    
    methods (Access=protected)
        function [dataset_coded] = do_code(obj,dataset_plain)
            samples_coded = zeros(dataset_plain.samples_count,obj.one_sample_coded.features_count);
            
            for i = 1:dataset_plain.samples_count
                samples_coded(i,:) = transforms.sparse.gdmp.matching_pursuit(obj.sparse_dict,dataset_plain.samples(i,:),obj.coeffs_count)';
            end
            
            dataset_coded = dataset(dataset_plain.classes,samples_coded,dataset_plain.labels_idx);
        end
        
        function [dataset_plain_hat] = do_decode(obj,dataset_coded)
            samples_plain_hat = zeros(dataset_coded.samples_count,obj.one_sample_plain.features_count);
            
            for i = 1:dataset_coded.samples_count
                samples_plain_hat(i,:) = obj.sparse_dict * dataset_coded.samples(i,:)';
            end
            
            dataset_plain_hat = dataset(dataset_coded.classes,samples_plain_hat,dataset_coded.labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function [norm_dict] = normalize_dict(dict)
            norm_dict = dict ./ repmat(sqrt(sum(dict .^ 2,1)),size(dict,1),1);
        end
        
        function [dict] = dict_gradient_descent(initial_dict,samples,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count)
            dict = transforms.sparse.gdmp.normalize_dict(initial_dict);
            samples_transp = samples';
            
            for iter = 1:max_iter_count                
                coeffs = transforms.sparse.gdmp.matching_pursuit(dict,samples,coeffs_count);
                
                diff = (samples_transp - dict * coeffs);
                delta_dict = diff * coeffs';
                learning_rate = initial_learning_rate * (final_learning_rate / initial_learning_rate) ^ (iter / max_iter_count);
                
                dict = dict + learning_rate * delta_dict;
                dict = transforms.sparse.gdmp.normalize_dict(dict);
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
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.sparse.gdmp".\n');
            
            fprintf('  Proper construction.\n');
            
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp(s,4,1,1e-2,1e-4,20);
            
            assert(tc.check(size(t.sparse_dict) == [2 4]));
            assert(tc.matrix(t.sparse_dict) && tc.unitreal(abs(t.sparse_dict)));
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
            assert(t.initial_learning_rate == 1e-2);
            assert(t.final_learning_rate == 1e-4);
            assert(t.max_iter_count == 20);
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
            assert(tc.check(size(t.one_sample_coded.samples) == [1 4]));
            assert(tc.matrix(t.one_sample_coded.samples) && tc.number(t.one_sample_coded.samples));
            assert(tc.check(t.one_sample_coded.labels_idx == c(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 4);
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp(s,3,1,1e-2,1e-4,20);            
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
                title('GDMP transformed samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            fprintf('    With one kept coefficient.\n');
            
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp(s,3,1,1e-2,1e-4,20);            
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
                title('GDMP transformed samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'o','b');
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    With 2 kept coefficients.\n');
            
            A = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([ 0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)];
            c = ones(600,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.sparse.gdmp(s,3,2,1e-2,1e-4,20);            
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
                title('GDMP transformed samples.');
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
