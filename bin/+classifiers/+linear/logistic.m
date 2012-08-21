classdef logistic < classifier
    properties (GetAccess=public,SetAccess=immutable)
        classifiers_count;
        saved_class_pair;
        method_code;
        model_weights;
        problem_form;
        reg_type;
        reg_param;
        multiclass_form;
        train_num_workers;
        classify_num_workers;
    end
   
    methods (Access=public)
        function [obj] = logistic(train_sample,class_info,problem_form,reg_type,reg_param,multiclass_form,num_workers)
            assert(check.dataset_record(train_sample));
            assert(issparse(train_sample));
            assert(check.scalar(class_info));
            assert(check.classifier_info(class_info));
            assert(check.scalar(problem_form));
            assert(check.string(problem_form));
            assert(check.one_of(problem_form,'Primal','Dual'));
            assert(check.scalar(reg_type));
            assert(check.string(reg_type));
            assert((check.same(problem_form,'Primal') && check.one_of(reg_type,'L1','L2')) || ...
                   (check.same(problem_form,'Dual') && check.one_of(reg_type,'L2')));
            assert(check.scalar(reg_param));
            assert(check.number(reg_param));
            assert(reg_param > 0);
            assert(check.scalar(multiclass_form));
            assert(check.string(multiclass_form));
            assert(check.one_of(multiclass_form,'1va','1v1'));
            assert(check.scalar(num_workers) || (check.vector(num_workers) && (length(num_workers) == 2)));
            assert(check.natural(num_workers));
            assert(check.checkv(num_workers >= 1));
            assert(class_info.compatible(train_sample));
            
            if class_info.labels_count == 2
                classifiers_count_t = 1;
                saved_class_pair_t = [1 2];
            elseif check.same(multiclass_form,'1va')
                classifiers_count_t = class_info.labels_count;
                saved_class_pair_t = zeros(classifiers_count_t,2);
                
                for ii = 1:class_info.labels_count
                    saved_class_pair_t(ii,1) = ii;
                    saved_class_pair_t(ii,2) = 0;
                end
            else
                classifiers_count_t = class_info.labels_count * (class_info.labels_count - 1) / 2;
                saved_class_pair_t = zeros(classifiers_count_t,2);
                class_pair_idx = 1;
                
                for ii = 1:class_info.labels_count
                    for jj = (ii + 1):class_info.labels_count
                        saved_class_pair_t(class_pair_idx,1) = ii;
                        saved_class_pair_t(class_pair_idx,2) = jj;
                        class_pair_idx = class_pair_idx + 1;
                    end
                end
            end
            
            if check.same(problem_form,'Primal')
                if check.same(reg_type,'L1')
                    method_code_t = 6;
                else
                    method_code_t = 0;
                end
            else
                method_code_t = 7;
            end
            
            if length(num_workers) == 2
                train_num_workers_t = num_workers(1);
                classify_num_workers_t = num_workers(2);
            else
                train_num_workers_t = num_workers;
                classify_num_workers_t = num_workers;
            end
            
            if class_info.labels_count == 2
                model_weights_t = xtern.x_classifiers_liblinear_train_one_vs_one(train_sample,class_info,method_code_t,reg_param,train_num_workers_t);
            elseif check.same(multiclass_form,'1va')
                model_weights_t = xtern.x_classifiers_liblinear_train_one_vs_all(train_sample,class_info,method_code_t,reg_param,train_num_workers_t);
            else
                model_weights_t = xtern.x_classifiers_liblinear_train_one_vs_one(train_sample,class_info,method_code_t,reg_param,train_num_workers_t);
            end
            
            input_geometry = dataset.geometry(train_sample);
            
            obj = obj@classifier(input_geometry,class_info.labels);
            obj.classifiers_count = classifiers_count_t;
            obj.saved_class_pair = saved_class_pair_t;
            obj.method_code = method_code_t;
            obj.model_weights = model_weights_t;
            obj.problem_form = problem_form;
            obj.reg_type = reg_type;
            obj.reg_param = reg_param;
            obj.multiclass_form = multiclass_form;
            obj.train_num_workers = train_num_workers_t;
            obj.classify_num_workers = classify_num_workers_t;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample)
            N = dataset.count(sample);
            
            classifiers_decisions = xtern.x_classifiers_liblinear_classify(sample,obj.model_weights,obj.method_code,obj.reg_param,obj.classify_num_workers);
            
            if obj.saved_labels_count == 2
                classifiers_probs_t1 = 1 ./ (1 + 2.71828183 .^ (-classifiers_decisions));
                classifiers_probs = [classifiers_probs_t1; 1 - classifiers_probs_t1];
                
                [~,max_probs_idx] = max(classifiers_probs,[],1);
                
                labels_idx_hat = max_probs_idx;
                labels_confidence = bsxfun(@rdivide,classifiers_probs,sum(classifiers_probs,1));
            elseif check.same(obj.multiclass_form,'1va')
                classifiers_probs = 1 ./ (1 + 2.71828183 .^ (-classifiers_decisions));
            
                [~,max_probs_idx] = max(classifiers_probs,[],1);

                labels_idx_hat = max_probs_idx;
                labels_confidence = bsxfun(@rdivide,classifiers_probs,sum(classifiers_probs,1));
            else
                classifiers_probs = 1 ./ (1 + 2.71828183 .^ (-classifiers_decisions));
                pair_labels_idx = (classifiers_probs < 0.5) + 1;
                full_probs = zeros(obj.saved_labels_count,N);
                
                for ii = 1:N
                    for jj = 1:obj.classifiers_count
                        target_class = obj.saved_class_pair(jj,pair_labels_idx(jj,ii));
                        if pair_labels_idx(jj,ii) == 1
                            full_probs(target_class,ii) = full_probs(target_class,ii) + classifiers_probs(jj,ii);
                        else
                            full_probs(target_class,ii) = full_probs(target_class,ii) + (1 - classifiers_probs(jj,ii));
                        end
                    end
                end
                
                [~,max_probs_idx] = max(full_probs,[],1);

                labels_idx_hat = max_probs_idx;
                labels_confidence = bsxfun(@rdivide,full_probs,sum(full_probs,1));
            end
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "classifiers.linear.logistic".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    In primal form, L1 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.linear.logistic(s,ci,'Primal','L1',1,'1va',1);

            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(cl.method_code == 6);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 3]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,201:300));
            assert(check.same(cl.problem_form,'Primal'));
            assert(check.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_workers == 1);
            assert(cl.classify_num_workers == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            clearvars -except test_figure;
            
            fprintf('    In primal form, L1 regularization and One-vs-One multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.linear.logistic(s,ci,'Primal','L1',1,'1v1',1);

            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(cl.method_code == 6);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 3]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,201:300));
            assert(check.same(cl.problem_form,'Primal'));
            assert(check.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_workers == 1);
            assert(cl.classify_num_workers == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            clearvars -except test_figure;
            
            fprintf('    In primal form, L2 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.linear.logistic(s,ci,'Primal','L2',1,'1va',1);

            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(cl.method_code == 0);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 3]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,201:300));
            assert(check.same(cl.problem_form,'Primal'));
            assert(check.same(cl.reg_type,'L2'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_workers == 1);
            assert(cl.classify_num_workers == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            clearvars -except test_figure;
            
            fprintf('    In primal form, L2 regularization and One-vs-One multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.linear.logistic(s,ci,'Primal','L2',1,'1v1',1);

            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(cl.method_code == 0);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 3]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,201:300));
            assert(check.same(cl.problem_form,'Primal'));
            assert(check.same(cl.reg_type,'L2'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_workers == 1);
            assert(cl.classify_num_workers == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            clearvars -except test_figure;
            
            fprintf('    In dual form, L2 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.linear.logistic(s,ci,'Dual','L2',1,'1va',1);

            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(cl.method_code == 7);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 3]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,201:300));
            assert(check.same(cl.problem_form,'Dual'));
            assert(check.same(cl.reg_type,'L2'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_workers == 1);
            assert(cl.classify_num_workers == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            clearvars -except test_figure;

            fprintf('    In dual form, L2 regularization and One-vs-One multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.linear.logistic(s,ci,'Dual','L2',1,'1v1',1);

            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(cl.method_code == 7);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 3]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,201:300));
            assert(check.same(cl.problem_form,'Dual'));
            assert(check.same(cl.reg_type,'L2'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_workers == 1);
            assert(cl.classify_num_workers == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            clearvars -except test_figure;
            
            fprintf('    With multiple threads and One-vs-Experiment multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.linear.logistic(s,ci,'Primal','L1',1,'1va',3);

            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(cl.method_code == 6);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 3]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,201:300));
            assert(check.same(cl.problem_form,'Primal'));
            assert(check.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_workers == 3);
            assert(cl.classify_num_workers == 3);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            clearvars -except test_figure;
            
            fprintf('    With multiple threads and One-vs-One multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.linear.logistic(s,ci,'Primal','L1',1,'1v1',3);

            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(cl.method_code == 6);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 3]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,201:300));
            assert(check.same(cl.problem_form,'Primal'));
            assert(check.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_workers == 3);
            assert(cl.classify_num_workers == 3);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            clearvars -except test_figure;
            
            fprintf('    With multiple train and classify threads and One-vs-Experiment multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.linear.logistic(s,ci,'Primal','L1',1,'1va',[3 2]);

            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(cl.method_code == 6);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 3]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,201:300));
            assert(check.same(cl.problem_form,'Primal'));
            assert(check.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_workers == 3);
            assert(cl.classify_num_workers == 2);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            clearvars -except test_figure;
            
            fprintf('    With multiple train and classify threads and One-vs-One multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_3.mat');
            
            cl = classifiers.linear.logistic(s,ci,'Primal','L1',1,'1v1',[3 2]);

            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(cl.method_code == 6);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 3]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,101:200));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,201:300));
            assert(check.same(cl.problem_form,'Primal'));
            assert(check.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_workers == 3);
            assert(cl.classify_num_workers == 2);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            clearvars -except test_figure;
            
            fprintf('    With two classes and One-vs-Experiment multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_2.mat');

            cl = classifiers.linear.logistic(s,ci,'Primal','L1',1,'1va',1);

            assert(cl.classifiers_count == 1);
            assert(check.same(cl.saved_class_pair,[1 2]));
            assert(cl.method_code == 6);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 1]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.same(cl.problem_form,'Primal'));
            assert(check.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_workers == 1);
            assert(cl.classify_num_workers == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2'}));
            assert(cl.saved_labels_count == 2);
            
            clearvars -except test_figure;
            
            fprintf('    With two classes and One-vs-One multiclass handling.\n');
            
            [s,ci] = dataset.load('../test/classifier_data_2.mat');

            cl = classifiers.linear.logistic(s,ci,'Primal','L1',1,'1v1',1);

            assert(cl.classifiers_count == 1);
            assert(check.same(cl.saved_class_pair,[1 2]));
            assert(cl.method_code == 6);
            assert(check.matrix(cl.model_weights));
            assert(check.same(size(cl.model_weights),[3 1]));
            assert(check.number(cl.model_weights));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(check.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(check.same(cl.problem_form,'Primal'));
            assert(check.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_workers == 1);
            assert(cl.classify_num_workers == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2'}));
            assert(cl.saved_labels_count == 2);
            
            clearvars -except test_figure;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    On clearly separated data.\n');
            
            fprintf('      In primal form, L1 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L1',1,'1va',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
	        assert(check.same(size(labels_confidence),[3 60]));
	        assert(check.unitreal(labels_confidence));
	        assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(check.empty(misclassified));

            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L1 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L1',1,'1v1',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
	        assert(check.same(size(labels_confidence),[3 60]));
	        assert(check.unitreal(labels_confidence));
	        assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(check.empty(misclassified));

            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L2 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L2',1,'1va',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
	        assert(check.same(size(labels_confidence),[3 60]));
	        assert(check.unitreal(labels_confidence));
	        assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(check.empty(misclassified));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L2 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L2',1,'1v1',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
	        assert(check.same(size(labels_confidence),[3 60]));
	        assert(check.unitreal(labels_confidence));
	        assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(check.empty(misclassified));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In dual form, L2 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Dual','L2',1,'1va',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
	        assert(check.same(size(labels_confidence),[3 60]));
	        assert(check.unitreal(labels_confidence));
	        assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(check.empty(misclassified));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In dual form, L2 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Dual','L2',1,'1v1',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
	        assert(check.same(size(labels_confidence),[3 60]));
	        assert(check.unitreal(labels_confidence));
	        assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(check.checkv(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(check.empty(misclassified));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    On clearly separated data with only two classes.\n');
            
            fprintf('      In primal form, L1 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_2.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_2.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L1',1,'1va',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[2 40]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,40)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
	        assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0; 0 20]));
            assert(check.empty(misclassified));

            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L1 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_2.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_2.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L1',1,'1v1',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[2 40]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,40)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
	        assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0; 0 20]));
            assert(check.empty(misclassified));

            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L2 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_2.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_2.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L2',1,'1va',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[2 40]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,40)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
	        assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0; 0 20]));
            assert(check.empty(misclassified));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L2 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_2.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_2.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L2',1,'1v1',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[2 40]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,40)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
	        assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0; 0 20]));
            assert(check.empty(misclassified));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In dual form, L2 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_2.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_2.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Dual','L2',1,'1va',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[2 40]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,40)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
	        assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0; 0 20]));
            assert(check.empty(misclassified));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In dual form, L2 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_clear_data_2.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_clear_data_2.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Dual','L2',1,'1v1',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat,ci_ts.labels_idx));
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[2 40]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,40)));
            assert(check.checkv(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
	        assert(check.checkv(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(check.checkv(conf_matrix == [20 0; 0 20]));
            assert(check.empty(misclassified));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    On mostly separated data.\n');
            
            fprintf('      In primal form, L1 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_mostly_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_mostly_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L1',1,'1va',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(check.matrix(labels_confidence));
	        assert(check.same(size(labels_confidence),[3 60]));
	        assert(check.unitreal(labels_confidence));
	        assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(check.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(check.same(misclassified,[19 20 39 40 59 60]));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L1 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_mostly_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_mostly_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L1',1,'1v1',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[3 60]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(check.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(check.same(misclassified,[19 20 39 40 59 60]));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L2 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_mostly_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_mostly_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L2',1,'1va',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[3 60]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(check.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(check.same(misclassified,[19 20 39 40 59 60]));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L2 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_mostly_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_mostly_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L2',1,'1v1',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[3 60]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(check.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(check.same(misclassified,[19 20 39 40 59 60]));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In dual form, L2 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_mostly_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_mostly_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Dual','L2',1,'1va',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[3 60]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(check.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(check.same(misclassified,[19 20 39 40 59 60]));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In dual form, L2 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_mostly_clear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_mostly_clear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Dual','L2',1,'1v1',1);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(check.matrix(labels_confidence));
            assert(check.same(size(labels_confidence),[3 60]));
            assert(check.unitreal(labels_confidence));
            assert(check.same(sum(labels_confidence,1),ones(1,60)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(check.checkv(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(check.checkv(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(check.checkv(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(check.checkv(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(check.checkv(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(check.checkv(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(check.checkv(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(check.checkv(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(check.checkv(labels_confidence(2,60) >= labels_confidence(1,60)));
	        assert(check.checkv(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(check.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(check.same(misclassified,[19 20 39 40 59 60]));
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    On not so clearly separated data.\n');
            
            fprintf('      In primal form, L1 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_unclear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_unclear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L1',1,'1va',1);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L1 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_unclear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_unclear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L1',1,'1v1',1);
                                                      
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L2 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_unclear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_unclear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L2',1,'1va',1);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In primal form, L2 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_unclear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_unclear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Primal','L2',1,'1v1',1);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In dual form, L2 regularization and One-vs-Experiment multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_unclear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_unclear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Dual','L2',1,'1va',1);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('      In dual form, L2 regularization and One-vs-One multiclass handling.\n');
            
            [s_tr,ci_tr] = dataset.load('../test/classifier_unclear_data_3.train.mat');
            [s_ts,ci_ts] = dataset.load('../test/classifier_unclear_data_3.test.mat');
            
            cl = classifiers.linear.logistic(s_tr,ci_tr,'Dual','L2',1,'1v1',1);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            clearvars -except test_figure;
        end
    end
end
