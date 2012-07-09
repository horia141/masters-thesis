classdef svm_kernel < classifier
    properties (GetAccess=public,SetAccess=immutable)
        classifiers_count;
        saved_class_pair;
        support_vectors_count;
        support_vectors;
        coeffs;
        rhos;
        prob_as;
        prob_bs;
        kernel_code;
        kernel_param1;
        kernel_param2;
        kernel_type;
        kernel_param;
        reg_param;
        multiclass_form;
        train_num_threads;
        classify_num_threads;
    end
    
    methods (Access=public)
        function [obj] = svm_kernel(train_sample,class_info,kernel_type,kernel_param,reg_param,multiclass_form,num_threads,logger)
            assert(check.dataset_record(train_sample));
            assert(check.scalar(class_info));
            assert(check.classifier_info(class_info));
            assert(check.scalar(kernel_type));
            assert(check.string(kernel_type));
            assert(check.one_of(kernel_type,'Linear','Polynomial','Gaussian','Logistic'));
            assert(check.vector(kernel_param));
            assert(check.number(kernel_param));
            assert((check.same(kernel_type,'Linear') && check.same(kernel_param,0)) || ...
                   (check.same(kernel_type,'Polynomial') && (length(kernel_param) == 2) && ...
                                                         (check.natural(kernel_param(1))) && ...
                                                         (kernel_param(1) >= 2) && ...
                                                         (kernel_param(2) >= 0)) || ...
                   (check.same(kernel_type,'Gaussian') && (length(kernel_param) == 1) && ...
                                                       (kernel_param > 0)) || ...
                   (check.same(kernel_type,'Logistic') && (length(kernel_param) == 2) && ...
                                                      (kernel_param(1) > 0) && ...
                                                      (kernel_param(2) >= 0)));
            assert(check.scalar(reg_param));
            assert(check.number(reg_param));
            assert(reg_param > 0);
            assert(check.scalar(multiclass_form));
            assert(check.string(multiclass_form));
            assert(check.one_of(multiclass_form,'1va','1v1'));
            assert(check.scalar(num_threads) || (check.vector(num_threads) && (length(num_threads) == 2)));
            assert(check.natural(num_threads));
            assert(check.checkv(num_threads >= 1));
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
            assert(logger.active);
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
            
            if check.same(kernel_type,'Linear')
                kernel_code_t = 0;
                kernel_param1_t = 0;
                kernel_param2_t = 0;
            elseif check.same(kernel_type,'Polynomial')
                kernel_code_t = 1;
                kernel_param1_t = kernel_param(1);
                kernel_param2_t = kernel_param(2);
            elseif check.same(kernel_type,'Gaussian')
                kernel_code_t = 2;
                kernel_param1_t = kernel_param(1);
                kernel_param2_t = 0;
            else
                kernel_code_t = 3;
                kernel_param1_t = kernel_param(1);
                kernel_param2_t = kernel_param(2);
            end
            
            if length(num_threads) == 2
                train_num_threads_t = num_threads(1);
                classify_num_threads_t = num_threads(2);
            else
                train_num_threads_t = num_threads;
                classify_num_threads_t = num_threads;
            end
            
            if class_info.labels_count == 2
                [support_vectors_count_t,support_vectors_t,coeffs_t,rhos_t,prob_as_t,prob_bs_t] = classifiers.libsvm.x_do_train_one_vs_one(train_sample,class_info,kernel_code_t,kernel_param1_t,kernel_param2_t,reg_param,train_num_threads_t,logger.new_classifier('Training each classifier'));
            elseif check.same(multiclass_form,'1va')
                [support_vectors_count_t,support_vectors_t,coeffs_t,rhos_t,prob_as_t,prob_bs_t] = classifiers.libsvm.x_do_train_one_vs_all(train_sample,class_info,kernel_code_t,kernel_param1_t,kernel_param2_t,reg_param,train_num_threads_t,logger.new_classifier('Training each classifier'));
            else
                [support_vectors_count_t,support_vectors_t,coeffs_t,rhos_t,prob_as_t,prob_bs_t] = classifiers.libsvm.x_do_train_one_vs_one(train_sample,class_info,kernel_code_t,kernel_param1_t,kernel_param2_t,reg_param,train_num_threads_t,logger.new_classifier('Training each classifier'));
            end
            
            input_geometry = dataset.geometry(train_sample);
            
            obj = obj@classifier(input_geometry,class_info.labels,logger);
            obj.classifiers_count = classifiers_count_t;
            obj.saved_class_pair = saved_class_pair_t;
            obj.support_vectors_count = support_vectors_count_t;
            obj.support_vectors = support_vectors_t;
            obj.coeffs = coeffs_t;
            obj.rhos = rhos_t;
            obj.prob_as = prob_as_t;
            obj.prob_bs = prob_bs_t;
            obj.kernel_code = kernel_code_t;
            obj.kernel_param1 = kernel_param1_t;
            obj.kernel_param2 = kernel_param2_t;
            obj.kernel_type = kernel_type;
            obj.kernel_param = kernel_param;
            obj.reg_param = reg_param;
            obj.multiclass_form = multiclass_form;
            obj.train_num_threads = train_num_threads_t;
            obj.classify_num_threads = classify_num_threads_t;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            N = dataset.count(sample);
            
            classifiers_decisions = classifiers.libsvm.x_do_classify(sample,obj.support_vectors_count,obj.support_vectors,obj.coeffs,obj.rhos,obj.prob_as,obj.prob_bs,obj.kernel_code,obj.kernel_param1,obj.kernel_param2,obj.reg_param,obj.classify_num_threads,logger.new_classifier('Classifying with each classifier'));
            
            logger.message('Determining most probable class.');

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
                pair_labels_idx = (classifiers_decisions < 0) + 1;
                partial_labels_idx = zeros(N,obj.classifiers_count);

                for ii = 1:obj.classifiers_count
                    partial_labels_idx(:,ii) = obj.saved_class_pair(ii,pair_labels_idx(ii,:))';
                end

                votes = hist(partial_labels_idx',obj.saved_labels_count);
                [~,labels_idx_hat_t] = max(votes,[],1);

                labels_idx_hat = labels_idx_hat_t;
                labels_confidence = bsxfun(@rdivide,votes,sum(votes,1));
            end
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "classifiers.svm_kernel".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Linear kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',1,logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Linear'));
            assert(check.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.classify_num_threads == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;

            fprintf('    Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',1,logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Linear'));
            assert(check.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.classify_num_threads == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Polynomial kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Polynomial',[2 3.5],1,'1va',1,logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 1);
            assert(cl.kernel_param1 == 2);
            assert(cl.kernel_param2 == 3.5);
            assert(check.same(cl.kernel_type,'Polynomial'));
            assert(check.same(cl.kernel_param,[2 3.5]));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.classify_num_threads == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Polynomial',[2 3.5],1,'1v1',1,logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 1);
            assert(cl.kernel_param1 == 2);
            assert(cl.kernel_param2 == 3.5);
            assert(check.same(cl.kernel_type,'Polynomial'));
            assert(check.same(cl.kernel_param,[2 3.5]));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.classify_num_threads == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Gaussian kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Gaussian',3.4,1,'1va',1,logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 2);
            assert(cl.kernel_param1 == 3.4);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Gaussian'));
            assert(check.same(cl.kernel_param,3.4));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.classify_num_threads == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Gaussian',3.4,1,'1v1',1,logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 2);
            assert(cl.kernel_param1 == 3.4);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Gaussian'));
            assert(check.same(cl.kernel_param,3.4));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.classify_num_threads == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Logistic kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Logistic',[0.05 0],1,'1va',1,logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 3);
            assert(cl.kernel_param1 == 0.05);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Logistic'));
            assert(check.same(cl.kernel_param,[0.05 0]));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.classify_num_threads == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    Logistic kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Logistic',[0.05 0],1,'1v1',1,logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 3);
            assert(cl.kernel_param1 == 0.05);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Logistic'));
            assert(check.same(cl.kernel_param,[0.05 0]));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.classify_num_threads == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With multiple threads and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',3,logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Linear'));
            assert(check.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 3);
            assert(cl.classify_num_threads == 3);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With multiple threads and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',3,logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Linear'));
            assert(check.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 3);
            assert(cl.classify_num_threads == 3);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With multiple train and classify threads and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',[3 2],logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Linear'));
            assert(check.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 3);
            assert(cl.classify_num_threads == 2);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With multiple train and classify threads and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',[3 2],logg);
            
            assert(cl.classifiers_count == 3);
            assert(check.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(check.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(check.cell(cl.support_vectors_count));
            assert(check.checkf(@check.vector,cl.support_vectors_count));
            assert(check.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(check.checkf(@check.natural,cl.support_vectors_count));
            assert(check.checkf(@(a)check.checkv(a > 0),cl.support_vectors_count));
            assert(check.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(check.cell(cl.support_vectors));
            assert(check.checkf(@check.matrix,cl.support_vectors));
            assert(check.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(check.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.support_vectors));
            assert(check.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(check.cell(cl.coeffs));
            assert(check.checkf(@check.vector,cl.coeffs));
            assert(check.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(check.checkf(@check.number,cl.coeffs));
            assert(check.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(check.cell(cl.rhos));
            assert(check.checkf(@check.scalar,cl.rhos));
            assert(check.checkf(@check.number,cl.rhos));
            assert(check.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(check.cell(cl.prob_as));
            assert(check.checkf(@check.scalar,cl.prob_as));
            assert(check.checkf(@check.number,cl.prob_as));
            assert(check.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(check.cell(cl.prob_bs));
            assert(check.checkf(@check.scalar,cl.prob_bs));
            assert(check.checkf(@check.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Linear'));
            assert(check.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 3);
            assert(cl.classify_num_threads == 2);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With two classes and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_2();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',1,logg);
            
            assert(cl.classifiers_count == 1);
            assert(check.same(cl.saved_class_pair,[1 2]));
            assert(check.scalar(cl.support_vectors_count));
            assert(check.cell(cl.support_vectors_count));
            assert(check.vector(cl.support_vectors_count{1}));
            assert(length(cl.support_vectors_count{1}) == 2);
            assert(check.natural(cl.support_vectors_count{1}));
            assert(check.checkv(cl.support_vectors_count{1} > 0));
            assert(check.scalar(cl.support_vectors));
            assert(check.cell(cl.support_vectors));
            assert(check.matrix(cl.support_vectors{1}));
            assert(size(cl.support_vectors{1},1) == 2);
            assert(size(cl.support_vectors{1},2) == sum(cl.support_vectors_count{1}));
            assert(check.number(cl.support_vectors{1}));
            assert(check.scalar(cl.coeffs));
            assert(check.vector(cl.coeffs{1}));
            assert(length(cl.coeffs{1}) == sum(cl.support_vectors_count{1}));
            assert(check.number(cl.coeffs{1}));
            assert(check.scalar(cl.rhos));
            assert(check.cell(cl.rhos));
            assert(check.scalar(cl.rhos{1}));
            assert(check.number(cl.rhos{1}));
            assert(check.scalar(cl.prob_as));
            assert(check.cell(cl.prob_as));
            assert(check.scalar(cl.prob_as{1}));
            assert(check.number(cl.prob_as{1}));
            assert(check.scalar(cl.prob_bs));
            assert(check.cell(cl.prob_bs));
            assert(check.scalar(cl.prob_bs{1}));
            assert(check.number(cl.prob_bs{1}));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Linear'));
            assert(check.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.classify_num_threads == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2'}));
            assert(cl.saved_labels_count == 2);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With two classes and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s,ci] = utils.testing.classifier_data_2();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',1,logg);
            
            assert(cl.classifiers_count == 1);
            assert(check.same(cl.saved_class_pair,[1 2]));
            assert(check.scalar(cl.support_vectors_count));
            assert(check.cell(cl.support_vectors_count));
            assert(check.vector(cl.support_vectors_count{1}));
            assert(length(cl.support_vectors_count{1}) == 2);
            assert(check.natural(cl.support_vectors_count{1}));
            assert(check.checkv(cl.support_vectors_count{1} > 0));
            assert(check.scalar(cl.support_vectors));
            assert(check.cell(cl.support_vectors));
            assert(check.matrix(cl.support_vectors{1}));
            assert(size(cl.support_vectors{1},1) == 2);
            assert(size(cl.support_vectors{1},2) == sum(cl.support_vectors_count{1}));
            assert(check.number(cl.support_vectors{1}));
            assert(check.scalar(cl.coeffs));
            assert(check.vector(cl.coeffs{1}));
            assert(length(cl.coeffs{1}) == sum(cl.support_vectors_count{1}));
            assert(check.number(cl.coeffs{1}));
            assert(check.scalar(cl.rhos));
            assert(check.cell(cl.rhos));
            assert(check.scalar(cl.rhos{1}));
            assert(check.number(cl.rhos{1}));
            assert(check.scalar(cl.prob_as));
            assert(check.cell(cl.prob_as));
            assert(check.scalar(cl.prob_as{1}));
            assert(check.number(cl.prob_as{1}));
            assert(check.scalar(cl.prob_bs));
            assert(check.cell(cl.prob_bs));
            assert(check.scalar(cl.prob_bs{1}));
            assert(check.number(cl.prob_bs{1}));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(check.same(cl.kernel_type,'Linear'));
            assert(check.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(check.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.classify_num_threads == 1);
            assert(check.same(cl.input_geometry,2));
            assert(check.same(cl.saved_labels,{'1' '2'}));
            assert(cl.saved_labels_count == 2);
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    On clearly separated data.\n');
            
            fprintf('      Linear kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Polynomial kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Gaussian kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Logistic kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Logistic kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    On clearly separated data with only two classes.\n');
            
            fprintf('      Linear kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Polynomial kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Gaussian kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Logistic kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Logistic kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    On mostly separated data.\n');
            
            fprintf('      Linear kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Polynomial kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Gaussian kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Logistic kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Logistic kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1va',1,logg);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,logg);
            
            assert(check.same(labels_idx_hat(1:18),ci_ts.labels_idx(1:18)));
            assert(check.same(labels_idx_hat(21:38),ci_ts.labels_idx(21:38)));
            assert(check.same(labels_idx_hat(41:58),ci_ts.labels_idx(41:58)));
            assert(labels_idx_hat(19) == 2);
            assert(labels_idx_hat(20) == 3);
            assert(labels_idx_hat(39) == 1);
            assert(labels_idx_hat(40) == 3);
            assert(labels_idx_hat(59) == 1);
            assert(labels_idx_hat(60) == 2);
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
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    On not so clearly separated data.\n');
            
            fprintf('      Linear kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,logg);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,logg);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Polynomial kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,logg);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,logg);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Gaussian kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,logg);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,logg);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Logistic kernel and One-vs-Experiment multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1va',1,logg);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('      Logistic kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utils.testing.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1v1',1,logg);
            
            if test_figure ~= -1
                figure(test_figure);
                utils.display.classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
        end
    end
end
