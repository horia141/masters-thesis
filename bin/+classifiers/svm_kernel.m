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
        train_max_wait_seconds;
        classify_num_threads;
        classify_max_wait_seconds;
    end
    
    methods (Access=public)
        function [obj] = svm_kernel(train_sample,class_info,kernel_type,kernel_param,reg_param,multiclass_form,num_threads,max_wait_seconds,logger)
            assert(tc.dataset_record(train_sample));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
            assert(tc.scalar(kernel_type));
            assert(tc.string(kernel_type));
            assert(tc.one_of(kernel_type,'Linear','Polynomial','Gaussian','Logistic'));
            assert(tc.vector(kernel_param));
            assert(tc.number(kernel_param));
            assert((tc.same(kernel_type,'Linear') && tc.same(kernel_param,0)) || ...
                   (tc.same(kernel_type,'Polynomial') && (length(kernel_param) == 2) && ...
                                                         (tc.natural(kernel_param(1))) && ...
                                                         (kernel_param(1) >= 2) && ...
                                                         (kernel_param(2) >= 0)) || ...
                   (tc.same(kernel_type,'Gaussian') && (length(kernel_param) == 1) && ...
                                                       (kernel_param > 0)) || ...
                   (tc.same(kernel_type,'Logistic') && (length(kernel_param) == 2) && ...
                                                      (kernel_param(1) > 0) && ...
                                                      (kernel_param(2) >= 0)));
            assert(tc.scalar(reg_param));
            assert(tc.number(reg_param));
            assert(reg_param > 0);
            assert(tc.scalar(multiclass_form));
            assert(tc.string(multiclass_form));
            assert(tc.one_of(multiclass_form,'1va','1v1'));
            assert(tc.scalar(num_threads) || (tc.vector(num_threads) && (length(num_threads) == 2)));
            assert(tc.natural(num_threads));
            assert(tc.check(num_threads >= 1));
            assert(tc.scalar(max_wait_seconds) || (tc.vector(max_wait_seconds) && (length(max_wait_seconds) == 2)));
            assert(tc.natural(max_wait_seconds));
            assert(tc.check(max_wait_seconds >= 1));
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
            
            if tc.same(kernel_type,'Linear')
                kernel_code_t = 0;
                kernel_param1_t = 0;
                kernel_param2_t = 0;
            elseif tc.same(kernel_type,'Polynomial')
                kernel_code_t = 1;
                kernel_param1_t = kernel_param(1);
                kernel_param2_t = kernel_param(2);
            elseif tc.same(kernel_type,'Gaussian')
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
            
            if length(max_wait_seconds) == 2
                train_max_wait_seconds_t = max_wait_seconds(1);
                classify_max_wait_seconds_t = max_wait_seconds(2);
            else
                train_max_wait_seconds_t = max_wait_seconds;
                classify_max_wait_seconds_t = max_wait_seconds;
            end
            
            if class_info.labels_count == 2
                [support_vectors_count_t,support_vectors_t,coeffs_t,rhos_t,prob_as_t,prob_bs_t] = classifiers.libsvm.x_do_train_one_vs_one(train_sample,class_info,kernel_code_t,kernel_param1_t,kernel_param2_t,reg_param,train_num_threads_t,train_max_wait_seconds_t,logger.new_classifier('Training each classifier'));
            elseif tc.same(multiclass_form,'1va')
                [support_vectors_count_t,support_vectors_t,coeffs_t,rhos_t,prob_as_t,prob_bs_t] = classifiers.libsvm.x_do_train_one_vs_all(train_sample,class_info,kernel_code_t,kernel_param1_t,kernel_param2_t,reg_param,train_num_threads_t,train_max_wait_seconds_t,logger.new_classifier('Training each classifier'));
            else
                [support_vectors_count_t,support_vectors_t,coeffs_t,rhos_t,prob_as_t,prob_bs_t] = classifiers.libsvm.x_do_train_one_vs_one(train_sample,class_info,kernel_code_t,kernel_param1_t,kernel_param2_t,reg_param,train_num_threads_t,train_max_wait_seconds_t,logger.new_classifier('Training each classifier'));
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
            obj.train_max_wait_seconds = train_max_wait_seconds_t;
            obj.classify_num_threads = classify_num_threads_t;
            obj.classify_max_wait_seconds = classify_max_wait_seconds_t;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            N = dataset.count(sample);
            
            classifiers_decisions = classifiers.libsvm.x_do_classify(sample,obj.support_vectors_count,obj.support_vectors,obj.coeffs,obj.rhos,obj.prob_as,obj.prob_bs,obj.kernel_code,obj.kernel_param1,obj.kernel_param2,obj.reg_param,obj.classify_num_threads,obj.classify_max_wait_seconds,logger.new_classifier('Classifying with each classifier'));
            
            logger.message('Determining most probable class.');

%             if obj.saved_labels_count == 2
%                 classifiers_probs_t1 = [classifiers_probs; 1 - classifiers_probs];
%                 [~,max_probs_idx] = max(classifiers_probs_t1,[],1);
%                 
%                 labels_idx_hat = max_probs_idx;
%                 labels_confidence = bsxfun(@rdivide,classifiers_probs_t1,sum(classifiers_probs_t1,1));
%             elseif tc.same(obj.multiclass_form,'1va')
%                 [~,max_probs_idx] = max(classifiers_probs,[],1);
% 
%                 labels_idx_hat = max_probs_idx;
%                 labels_confidence = bsxfun(@rdivide,classifiers_probs,sum(classifiers_probs,1));
%             else
%                 pair_labels_idx = (classifiers_probs < 0.5) + 1;
%                 full_probs = zeros(obj.saved_labels_count,N);
%                 
%                 for ii = 1:N
%                     for jj = 1:obj.classifiers_count
%                         target_class = obj.saved_class_pair(jj,pair_labels_idx(jj,ii));
%                         if pair_labels_idx(jj,ii) == 1
%                             full_probs(target_class,ii) = full_probs(target_class,ii) + classifiers_probs(jj,ii);
%                         else
%                             full_probs(target_class,ii) = full_probs(target_class,ii) + (1 - classifiers_probs(jj,ii));
%                         end
%                     end
%                 end
%                 
%                 [~,max_probs_idx] = max(full_probs,[],1);
% 
%                 labels_idx_hat = max_probs_idx;
%                 labels_confidence = bsxfun(@rdivide,full_probs,sum(full_probs,1));
%             end

            if obj.saved_labels_count == 2
                classifiers_probs_t1 = 1 ./ (1 + 2.71828183 .^ (-classifiers_decisions));
                classifiers_probs = [classifiers_probs_t1; 1 - classifiers_probs_t1];
                
                [~,max_probs_idx] = max(classifiers_probs,[],1);
                
                labels_idx_hat = max_probs_idx;
                labels_confidence = bsxfun(@rdivide,classifiers_probs,sum(classifiers_probs,1));
            elseif tc.same(obj.multiclass_form,'1va')
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
        function test(display)
            fprintf('Testing "classifiers.svm_kernel".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Linear kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',1,3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;

            fprintf('    Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',1,3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Polynomial kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Polynomial',[2 3.5],1,'1va',1,3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 1);
            assert(cl.kernel_param1 == 2);
            assert(cl.kernel_param2 == 3.5);
            assert(tc.same(cl.kernel_type,'Polynomial'));
            assert(tc.same(cl.kernel_param,[2 3.5]));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Polynomial',[2 3.5],1,'1v1',1,3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 1);
            assert(cl.kernel_param1 == 2);
            assert(cl.kernel_param2 == 3.5);
            assert(tc.same(cl.kernel_type,'Polynomial'));
            assert(tc.same(cl.kernel_param,[2 3.5]));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Gaussian kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Gaussian',3.4,1,'1va',1,3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 2);
            assert(cl.kernel_param1 == 3.4);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Gaussian'));
            assert(tc.same(cl.kernel_param,3.4));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Gaussian',3.4,1,'1v1',1,3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 2);
            assert(cl.kernel_param1 == 3.4);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Gaussian'));
            assert(tc.same(cl.kernel_param,3.4));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Logistic kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Logistic',[0.05 0],1,'1va',1,3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 3);
            assert(cl.kernel_param1 == 0.05);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Logistic'));
            assert(tc.same(cl.kernel_param,[0.05 0]));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Logistic kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Logistic',[0.05 0],1,'1v1',1,3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 3);
            assert(cl.kernel_param1 == 0.05);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Logistic'));
            assert(tc.same(cl.kernel_param,[0.05 0]));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
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
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',3,3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 3);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 3);
            assert(cl.classify_max_wait_seconds == 3);
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
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',3,3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 3);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 3);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With multiple train and classify threads and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',[3 2],3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 3);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 2);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With multiple train and classify threads and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',[3 2],3,log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 3);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 2);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With different train and classify wait times and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',1,[5 3],log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 0; 2 0; 3 0]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 5);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2' '3'}));
            assert(cl.saved_labels_count == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With different train and classify wait times and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',1,[5 3],log);
            
            assert(cl.classifiers_count == 3);
            assert(tc.same(cl.saved_class_pair,[1 2; 1 3; 2 3]));
            assert(tc.vector(cl.support_vectors_count));
            assert(length(cl.support_vectors_count) == 3);
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.checkf(@tc.vector,cl.support_vectors_count));
            assert(tc.checkf(@(a)length(a) == 2,cl.support_vectors_count));
            assert(tc.checkf(@tc.natural,cl.support_vectors_count));
            assert(tc.checkf(@(a)tc.check(a > 0),cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors));
            assert(length(cl.support_vectors) == 3);
            assert(tc.cell(cl.support_vectors));
            assert(tc.checkf(@tc.matrix,cl.support_vectors));
            assert(tc.checkf(@(a)size(a,1) == 2,cl.support_vectors));
            assert(tc.checkf(@(ii)size(cl.support_vectors{ii},2) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.support_vectors));
            assert(tc.vector(cl.coeffs));
            assert(length(cl.coeffs) == 3);
            assert(tc.cell(cl.coeffs));
            assert(tc.checkf(@tc.vector,cl.coeffs));
            assert(tc.checkf(@(ii)length(cl.coeffs{ii}) == sum(cl.support_vectors_count{ii}),1:3));
            assert(tc.checkf(@tc.number,cl.coeffs));
            assert(tc.vector(cl.rhos));
            assert(length(cl.rhos) == 3);
            assert(tc.cell(cl.rhos));
            assert(tc.checkf(@tc.scalar,cl.rhos));
            assert(tc.checkf(@tc.number,cl.rhos));
            assert(tc.vector(cl.prob_as));
            assert(length(cl.prob_as) == 3);
            assert(tc.cell(cl.prob_as));
            assert(tc.checkf(@tc.scalar,cl.prob_as));
            assert(tc.checkf(@tc.number,cl.prob_as));
            assert(tc.vector(cl.prob_bs));
            assert(length(cl.prob_bs) == 3);
            assert(tc.cell(cl.prob_bs));
            assert(tc.checkf(@tc.scalar,cl.prob_bs));
            assert(tc.checkf(@tc.number,cl.prob_bs));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 5);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
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
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',1,3,log);
            
            assert(cl.classifiers_count == 1);
            assert(tc.same(cl.saved_class_pair,[1 2]));
            assert(tc.scalar(cl.support_vectors_count));
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors_count{1}));
            assert(length(cl.support_vectors_count{1}) == 2);
            assert(tc.natural(cl.support_vectors_count{1}));
            assert(tc.check(cl.support_vectors_count{1} > 0));
            assert(tc.scalar(cl.support_vectors));
            assert(tc.cell(cl.support_vectors));
            assert(tc.matrix(cl.support_vectors{1}));
            assert(size(cl.support_vectors{1},1) == 2);
            assert(size(cl.support_vectors{1},2) == sum(cl.support_vectors_count{1}));
            assert(tc.number(cl.support_vectors{1}));
            assert(tc.scalar(cl.coeffs));
            assert(tc.vector(cl.coeffs{1}));
            assert(length(cl.coeffs{1}) == sum(cl.support_vectors_count{1}));
            assert(tc.number(cl.coeffs{1}));
            assert(tc.scalar(cl.rhos));
            assert(tc.cell(cl.rhos));
            assert(tc.scalar(cl.rhos{1}));
            assert(tc.number(cl.rhos{1}));
            assert(tc.scalar(cl.prob_as));
            assert(tc.cell(cl.prob_as));
            assert(tc.scalar(cl.prob_as{1}));
            assert(tc.number(cl.prob_as{1}));
            assert(tc.scalar(cl.prob_bs));
            assert(tc.cell(cl.prob_bs));
            assert(tc.scalar(cl.prob_bs{1}));
            assert(tc.number(cl.prob_bs{1}));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
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
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',1,3,log);
            
            assert(cl.classifiers_count == 1);
            assert(tc.same(cl.saved_class_pair,[1 2]));
            assert(tc.scalar(cl.support_vectors_count));
            assert(tc.cell(cl.support_vectors_count));
            assert(tc.vector(cl.support_vectors_count{1}));
            assert(length(cl.support_vectors_count{1}) == 2);
            assert(tc.natural(cl.support_vectors_count{1}));
            assert(tc.check(cl.support_vectors_count{1} > 0));
            assert(tc.scalar(cl.support_vectors));
            assert(tc.cell(cl.support_vectors));
            assert(tc.matrix(cl.support_vectors{1}));
            assert(size(cl.support_vectors{1},1) == 2);
            assert(size(cl.support_vectors{1},2) == sum(cl.support_vectors_count{1}));
            assert(tc.number(cl.support_vectors{1}));
            assert(tc.scalar(cl.coeffs));
            assert(tc.vector(cl.coeffs{1}));
            assert(length(cl.coeffs{1}) == sum(cl.support_vectors_count{1}));
            assert(tc.number(cl.coeffs{1}));
            assert(tc.scalar(cl.rhos));
            assert(tc.cell(cl.rhos));
            assert(tc.scalar(cl.rhos{1}));
            assert(tc.number(cl.rhos{1}));
            assert(tc.scalar(cl.prob_as));
            assert(tc.cell(cl.prob_as));
            assert(tc.scalar(cl.prob_as{1}));
            assert(tc.number(cl.prob_as{1}));
            assert(tc.scalar(cl.prob_bs));
            assert(tc.cell(cl.prob_bs));
            assert(tc.scalar(cl.prob_bs{1}));
            assert(tc.number(cl.prob_bs{1}));
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.train_num_threads == 1);
            assert(cl.train_max_wait_seconds == 3);
            assert(cl.classify_num_threads == 1);
            assert(cl.classify_max_wait_seconds == 3);
            assert(tc.same(cl.input_geometry,2));
            assert(tc.same(cl.saved_labels,{'1' '2'}));
            assert(cl.saved_labels_count == 2);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    On clearly separated data.\n');
            
            fprintf('      Linear kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Polynomial kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Gaussian kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Logistic kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1va',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0 0; 0 20 0; 0 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Logistic kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1v1',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[3 60]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(3,1:20)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(2,21:40)));
            assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(3,21:40)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(2,41:60)));
            assert(tc.check(labels_confidence(3,41:60) >= labels_confidence(3,41:60)));
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
            
            fprintf('      Linear kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,40)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
                assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,40)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
                assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Polynomial kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,40)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
                assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,40)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
                assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Gaussian kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,40)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
                assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,40)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
                assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Logistic kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1va',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,40)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
                assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
            assert(score == 100);
            assert(tc.check(conf_matrix == [20 0; 0 20]));
            assert(tc.empty(misclassified));

            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Logistic kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1v1',1,3,log);
            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = cl.classify(s_ts,ci_ts,log);
            
            assert(tc.same(labels_idx_hat,ci_ts.labels_idx));
            assert(tc.matrix(labels_confidence));
            assert(tc.same(size(labels_confidence),[2 40]));
            assert(tc.unitreal(labels_confidence));
            assert(tc.same(sum(labels_confidence,1),ones(1,40)));
            assert(tc.check(labels_confidence(1,1:20) >= labels_confidence(2,1:20)));
                assert(tc.check(labels_confidence(2,21:40) >= labels_confidence(1,21:40)));
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
            
            fprintf('      Linear kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,3,log);
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
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,3,log);
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
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Polynomial kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,3,log);
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
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,3,log);
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
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Gaussian kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,3,log);
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
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,3,log);
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
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Logistic kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1va',1,3,log);
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
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(3,60)));
            assert(score == 90);
            assert(tc.same(conf_matrix,[18 1 1; 1 18 1; 1 1 18]));
            assert(tc.same(misclassified,[19 20 39 40 59 60]));
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Logistic kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1va',1,3,log);
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
            assert(tc.same(sum(labels_confidence,1),ones(1,60)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(2,1:18)));
            assert(tc.check(labels_confidence(1,1:18) >= labels_confidence(3,1:18)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(1,19)));
            assert(tc.check(labels_confidence(2,19) >= labels_confidence(3,19)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(1,20)));
            assert(tc.check(labels_confidence(3,20) >= labels_confidence(2,20)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(2,21:38)));
            assert(tc.check(labels_confidence(2,21:38) >= labels_confidence(3,21:38)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(2,39)));
            assert(tc.check(labels_confidence(1,39) >= labels_confidence(3,39)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(1,40)));
            assert(tc.check(labels_confidence(3,40) >= labels_confidence(2,40)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(2,41:58)));
            assert(tc.check(labels_confidence(3,41:58) >= labels_confidence(3,41:58)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(2,59)));
            assert(tc.check(labels_confidence(1,59) >= labels_confidence(3,59)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(1,60)));
            assert(tc.check(labels_confidence(2,60) >= labels_confidence(3,60)));
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
            
            fprintf('      Linear kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,3,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,3,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Polynomial kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,3,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,3,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Gaussian kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,3,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,3,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Logistic kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1va',1,3,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Logistic kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Logistic',[0.05 0],1,'1v1',1,3,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
