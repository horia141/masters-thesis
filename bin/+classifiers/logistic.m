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
        num_threads;
    end
    
    methods (Access=public)
        function [obj] = logistic(train_sample,class_info,problem_form,reg_type,reg_param,multiclass_form,num_threads,logger)
            assert(tc.dataset_record(train_sample));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
            assert(tc.scalar(problem_form));
            assert(tc.string(problem_form));
            assert(tc.one_of(problem_form,'Primal','Dual'));
            assert(tc.scalar(reg_type));
            assert(tc.string(reg_type));
            assert((tc.same(problem_form,'Primal') && tc.one_of(reg_type,'L1','L2')) || ...
                   (tc.same(problem_form,'Dual') && tc.one_of(reg_type,'L2')));
            assert(tc.scalar(reg_param));
            assert(tc.number(reg_param));
            assert(reg_param > 0);
            assert(tc.scalar(multiclass_form));
            assert(tc.string(multiclass_form));
            assert(tc.one_of(multiclass_form,'1va','1v1'));
            assert(tc.scalar(num_threads));
            assert(tc.natural(num_threads));
            assert(num_threads >= 1);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(class_info.compatible(train_sample));
            
            if class_info.labels_count == 2
                classifiers_count_t = 1;
                saved_class_pair_t = [1 2];
            elseif tc.same(multiclass_form,'1va')
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
            
            if tc.same(problem_form,'Primal')
                if tc.same(reg_type,'L1')
                    method_code_t = 6;
                else
                    method_code_t = 0;
                end
            else
                method_code_t = 7;
            end
            
            if class_info.labels_count == 2
                model_weights_t = classifiers.liblinear.x_do_train_one_vs_one(train_sample,class_info,method_code_t,reg_param,num_threads,logger.new_classifier('Training each classifier'));
            elseif tc.same(multiclass_form,'1va')
                model_weights_t = classifiers.liblinear.x_do_train_one_vs_all(train_sample,class_info,method_code_t,reg_param,num_threads,logger.new_classifier('Training each classifier'));
            else
                model_weights_t = classifiers.liblinear.x_do_train_one_vs_one(train_sample,class_info,method_code_t,reg_param,num_threads,logger.new_classifier('Training each classifier'));
            end
            
            input_geometry = dataset.geometry(train_sample);
            
            obj = obj@classifier(input_geometry,class_info.labels,logger);
            obj.classifiers_count = classifiers_count_t;
            obj.saved_class_pair = saved_class_pair_t;
            obj.method_code = method_code_t;
            obj.model_weights = model_weights_t;
            obj.problem_form = problem_form;
            obj.reg_type = reg_type;
            obj.reg_param = reg_param;
            obj.multiclass_form = multiclass_form;
            obj.num_threads = num_threads;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            N = dataset.count(sample);
            
            classifiers_decisions = classifiers.liblinear.x_do_classify(sample,obj.model_weights,obj.method_code,obj.reg_param,logger.new_classifier('Classifying with each classifier'));
            
            logger.message('Determining most probable class.');
            
            if obj.saved_labels_count == 2
                classifiers_probs_t1 = 1 ./ (1 + 2.71314 .^ (-classifiers_decisions));
                classifiers_probs = [classifiers_probs_t1; 1 - classifiers_probs_t1];
                
                [max_probs,max_probs_idx] = max(classifiers_probs,[],1);
                
                labels_idx_hat = max_probs_idx;
                labels_confidence = bsxfun(@rdivide,classifiers_probs,max_probs);
            elseif tc.same(obj.multiclass_form,'1va')
                classifiers_probs = 1 ./ (1 + 2.718281 .^ (-classifiers_decisions));
            
                [max_probs,max_probs_idx] = max(classifiers_probs,[],1);

                labels_idx_hat = max_probs_idx;
                labels_confidence = bsxfun(@rdivide,classifiers_probs,max_probs);
            else
                classifiers_probs = 1 ./ (1 + 2.718281 .^ (-classifiers_decisions));
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
                
                [max_probs,max_probs_idx] = max(full_probs,[],1);

                labels_idx_hat = max_probs_idx;
                labels_confidence = bsxfun(@rdivide,full_probs,max_probs);
            end
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "classifiers.logistic".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    In primal form, L1 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.logistic(s,ci,'Primal','L1',1,'1va',1,log);

            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(cl.method_code == 6);
            assert(tc.matrix(cl.model_weights));
            assert(tc.same(size(cl.model_weights),[3 3]));
            assert(tc.number(cl.model_weights));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,201:300));
            assert(tc.same(cl.problem_form,'Primal'));
            assert(tc.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 1);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    In primal form, L1 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.logistic(s,ci,'Primal','L1',1,'1v1',1,log);

            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(cl.method_code == 6);
            assert(tc.matrix(cl.model_weights));
            assert(tc.same(size(cl.model_weights),[3 3]));
            assert(tc.number(cl.model_weights));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,201:300));
            assert(tc.same(cl.problem_form,'Primal'));
            assert(tc.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 1);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    In primal form, L2 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.logistic(s,ci,'Primal','L2',1,'1va',1,log);

            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(cl.method_code == 0);
            assert(tc.matrix(cl.model_weights));
            assert(tc.same(size(cl.model_weights),[3 3]));
            assert(tc.number(cl.model_weights));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,201:300));
            assert(tc.same(cl.problem_form,'Primal'));
            assert(tc.same(cl.reg_type,'L2'));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 1);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    In primal form, L2 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.logistic(s,ci,'Primal','L2',1,'1v1',1,log);

            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(cl.method_code == 0);
            assert(tc.matrix(cl.model_weights));
            assert(tc.same(size(cl.model_weights),[3 3]));
            assert(tc.number(cl.model_weights));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,201:300));
            assert(tc.same(cl.problem_form,'Primal'));
            assert(tc.same(cl.reg_type,'L2'));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 1);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    In dual form, L2 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.logistic(s,ci,'Dual','L2',1,'1va',1,log);

            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(cl.method_code == 7);
            assert(tc.matrix(cl.model_weights));
            assert(tc.same(size(cl.model_weights),[3 3]));
            assert(tc.number(cl.model_weights));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,201:300));
            assert(tc.same(cl.problem_form,'Dual'));
            assert(tc.same(cl.reg_type,'L2'));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 1);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            log.close();
            hnd.close();
            
            clearvars -except display;

            fprintf('    In dual form, L2 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.logistic(s,ci,'Dual','L2',1,'1v1',1,log);

            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(cl.method_code == 7);
            assert(tc.matrix(cl.model_weights));
            assert(tc.same(size(cl.model_weights),[3 3]));
            assert(tc.number(cl.model_weights));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,201:300));
            assert(tc.same(cl.problem_form,'Dual'));
            assert(tc.same(cl.reg_type,'L2'));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 1);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With multiple threads and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.logistic(s,ci,'Primal','L1',1,'1va',3,log);

            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(cl.method_code == 6);
            assert(tc.matrix(cl.model_weights));
            assert(tc.same(size(cl.model_weights),[3 3]));
            assert(tc.number(cl.model_weights));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,201:300));
            assert(tc.same(cl.problem_form,'Primal'));
            assert(tc.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With multiple threads and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.logistic(s,ci,'Primal','L1',1,'1v1',3,log);

            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(cl.method_code == 6);
            assert(tc.matrix(cl.model_weights));
            assert(tc.same(size(cl.model_weights),[3 3]));
            assert(tc.number(cl.model_weights));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,2) < 0,201:300));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) >= 0,101:200));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,3) < 0,201:300));
            assert(tc.same(cl.problem_form,'Primal'));
            assert(tc.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With two classes and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_2();

            cl = classifiers.logistic(s,ci,'Primal','L1',1,'1va',1,log);

            assert(cl.classifiers_count == 1);
            assert(tc.same(cl.saved_class_pair,[1 2]));
            assert(cl.method_code == 6);
            assert(tc.matrix(cl.model_weights));
            assert(tc.same(size(cl.model_weights),[3 1]));
            assert(tc.number(cl.model_weights));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(tc.same(cl.problem_form,'Primal'));
            assert(tc.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 1);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2'}));
            assert(cl.saved_labels_count == 2);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With two classes and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_2();

            cl = classifiers.logistic(s,ci,'Primal','L1',1,'1v1',1,log);

            assert(cl.classifiers_count == 1);
            assert(tc.same(cl.saved_class_pair,[1 2]));
            assert(cl.method_code == 6);
            assert(tc.matrix(cl.model_weights));
            assert(tc.same(size(cl.model_weights),[3 1]));
            assert(tc.number(cl.model_weights));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) >= 0,1:100));
            assert(tc.checkf(@(ii)[s(:,ii);1]' * cl.model_weights(:,1) < 0,101:200));
            assert(tc.same(cl.problem_form,'Primal'));
            assert(tc.same(cl.reg_type,'L1'));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 1);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2'}));
            assert(cl.saved_labels_count == 2);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    On clearly separated data.\n');
            
            fprintf('      In primal form, L1 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L1',1,'1va',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(tc.same(labels_confidence(3,41:60),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L1 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L1',1,'1v1',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(tc.same(labels_confidence(3,41:60),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L2 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L2',1,'1va',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(tc.same(labels_confidence(3,41:60),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L2 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L2',1,'1v1',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(tc.same(labels_confidence(3,41:60),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In dual form, L2 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Dual','L2',1,'1va',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(tc.same(labels_confidence(3,41:60),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In dual form, L2 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Dual','L2',1,'1v1',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(tc.same(labels_confidence(3,41:60),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            
            fprintf('    On clearly separated data with only two classes.\n');
            
            fprintf('      In primal form, L1 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L1',1,'1va',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L1 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L1',1,'1v1',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L2 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L2',1,'1va',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L2 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L2',1,'1v1',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In dual form, L2 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Dual','L2',1,'1va',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In dual form, L2 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Dual','L2',1,'1v1',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(labels_confidence(1,1:20),ones(1,20)));
            assert(tc.same(labels_confidence(2,21:40),ones(1,20)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    On mostly separated data.\n');
            
            fprintf('      In primal form, L1 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L1',1,'1va',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(tc.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(tc.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(tc.same(labels_confidence(1,1:18),ones(1,18)));
            assert(tc.same(labels_confidence(2,19),1));
            assert(tc.same(labels_confidence(3,20),1));
            assert(tc.same(labels_confidence(2,21:38),ones(1,18)));
            assert(tc.same(labels_confidence(1,39),1));
            assert(tc.same(labels_confidence(3,40),1));
            assert(tc.same(labels_confidence(3,41:58),ones(1,18)));
            assert(tc.same(labels_confidence(1,59),1));
            assert(tc.same(labels_confidence(2,60),1));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L1 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L1',1,'1v1',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(tc.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(tc.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(tc.same(labels_confidence(1,1:18),ones(1,18)));
            assert(tc.same(labels_confidence(2,19),1));
            assert(tc.same(labels_confidence(3,20),1));
            assert(tc.same(labels_confidence(2,21:38),ones(1,18)));
            assert(tc.same(labels_confidence(1,39),1));
            assert(tc.same(labels_confidence(3,40),1));
            assert(tc.same(labels_confidence(3,41:58),ones(1,18)));
            assert(tc.same(labels_confidence(1,59),1));
            assert(tc.same(labels_confidence(2,60),1));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L2 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L2',1,'1va',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(tc.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(tc.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(tc.same(labels_confidence(1,1:18),ones(1,18)));
            assert(tc.same(labels_confidence(2,19),1));
            assert(tc.same(labels_confidence(3,20),1));
            assert(tc.same(labels_confidence(2,21:38),ones(1,18)));
            assert(tc.same(labels_confidence(1,39),1));
            assert(tc.same(labels_confidence(3,40),1));
            assert(tc.same(labels_confidence(3,41:58),ones(1,18)));
            assert(tc.same(labels_confidence(1,59),1));
            assert(tc.same(labels_confidence(2,60),1));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L2 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L2',1,'1v1',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(tc.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(tc.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(tc.same(labels_confidence(1,1:18),ones(1,18)));
            assert(tc.same(labels_confidence(2,19),1));
            assert(tc.same(labels_confidence(3,20),1));
            assert(tc.same(labels_confidence(2,21:38),ones(1,18)));
            assert(tc.same(labels_confidence(1,39),1));
            assert(tc.same(labels_confidence(3,40),1));
            assert(tc.same(labels_confidence(3,41:58),ones(1,18)));
            assert(tc.same(labels_confidence(1,59),1));
            assert(tc.same(labels_confidence(2,60),1));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In dual form, L2 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Dual','L2',1,'1va',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(tc.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(tc.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(tc.same(labels_confidence(1,1:18),ones(1,18)));
            assert(tc.same(labels_confidence(2,19),1));
            assert(tc.same(labels_confidence(3,20),1));
            assert(tc.same(labels_confidence(2,21:38),ones(1,18)));
            assert(tc.same(labels_confidence(1,39),1));
            assert(tc.same(labels_confidence(3,40),1));
            assert(tc.same(labels_confidence(3,41:58),ones(1,18)));
            assert(tc.same(labels_confidence(1,59),1));
            assert(tc.same(labels_confidence(2,60),1));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In dual form, L2 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Dual','L2',1,'1v1',1,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(tc.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(tc.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
            assert(tc.same(labels_confidence(1,1:18),ones(1,18)));
            assert(tc.same(labels_confidence(2,19),1));
            assert(tc.same(labels_confidence(3,20),1));
            assert(tc.same(labels_confidence(2,21:38),ones(1,18)));
            assert(tc.same(labels_confidence(1,39),1));
            assert(tc.same(labels_confidence(3,40),1));
            assert(tc.same(labels_confidence(3,41:58),ones(1,18)));
            assert(tc.same(labels_confidence(1,59),1));
            assert(tc.same(labels_confidence(2,60),1));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    On not so clearly separated data.\n');
            
            fprintf('      In primal form, L1 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L1',1,'1va',1,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L1 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L1',1,'1v1',1,log);
                                                      
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L2 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L2',1,'1va',1,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In primal form, L2 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Primal','L2',1,'1v1',1,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In dual form, L2 regularization and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Dual','L2',1,'1va',1,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      In dual form, L2 regularization and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.logistic(s_tr,ci_tr,'Dual','L2',1,'1v1',1,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
