classdef svm_kernel < classifier
    properties (GetAccess=public,SetAccess=immutable)
        classifiers_count;
        saved_class_pair;
	    support_vectors_count;
	    support_vectors;
	    coeffs;
	    rhos;
	    kernel_code;
	    kernel_param1;
	    kernel_param2;
        kernel_type;
        kernel_param;
        reg_param;
        multiclass_form;
        num_threads;
    end
    
    methods (Access=public)
        function [obj] = svm_kernel(train_sample,class_info,kernel_type,kernel_param,reg_param,multiclass_form,num_threads,logger)
            assert(tc.dataset_record(train_sample));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
            assert(tc.scalar(kernel_type));
            assert(tc.string(kernel_type));
            assert(tc.one_of(kernel_type,'Linear','Polynomial','Gaussian','Sigmoid'));
            assert(tc.vector(kernel_param));
            assert(tc.number(kernel_param));
            assert((tc.same(kernel_type,'Linear') && tc.same(kernel_param,0)) || ...
                   (tc.same(kernel_type,'Polynomial') && (length(kernel_param) == 2) && ...
                                                         (tc.natural(kernel_param(1))) && ...
                                                         (kernel_param(1) >= 2) && ...
                                                         (kernel_param(2) >= 0)) || ...
                   (tc.same(kernel_type,'Gaussian') && (length(kernel_param) == 1) && ...
                                                       (kernel_param > 0)) || ...
                   (tc.same(kernel_type,'Sigmoid') && (length(kernel_param) == 2) && ...
                                                      (kernel_param(1) > 0) && ...
                                                      (kernel_param(2) >= 0)));
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
            
            if class_info.labels_count == 2
                [support_vectors_count_t,support_vectors_t,coeffs_t,rhos_t] = classifiers.libsvm.x_do_train_one_vs_one(train_sample,class_info,kernel_code_t,kernel_param1_t,kernel_param2_t,reg_param,num_threads,logger.new_classifier('Training each classifier'));
            elseif tc.same(multiclass_form,'1va')
                [support_vectors_count_t,support_vectors_t,coeffs_t,rhos_t] = classifiers.libsvm.x_do_train_one_vs_all(train_sample,class_info,kernel_code_t,kernel_param1_t,kernel_param2_t,reg_param,num_threads,logger.new_classifier('Training each classifier'));
            else
                [support_vectors_count_t,support_vectors_t,coeffs_t,rhos_t] = classifiers.libsvm.x_do_train_one_vs_one(train_sample,class_info,kernel_code_t,kernel_param1_t,kernel_param2_t,reg_param,num_threads,logger.new_classifier('Training each classifier'));
            end
            
            input_geometry = dataset.geometry(train_sample);
            
            obj = obj@classifier(input_geometry,class_info.labels,logger);
            obj.classifiers_count = classifiers_count_t;
            obj.saved_class_pair = saved_class_pair_t;
            obj.support_vectors_count = support_vectors_count_t;
            obj.support_vectors = support_vectors_t;
            obj.coeffs = coeffs_t;
            obj.rhos = rhos_t;
            obj.kernel_code = kernel_code_t;
            obj.kernel_param1 = kernel_param1_t;
            obj.kernel_param2 = kernel_param2_t;
            obj.kernel_type = kernel_type;
            obj.kernel_param = kernel_param;
            obj.reg_param = reg_param;
            obj.multiclass_form = multiclass_form;
            obj.num_threads = num_threads;
        end
    end
    
    methods (Access=protected)
        function [labels_idx_hat,labels_confidence] = do_classify(obj,sample,logger)
            N = dataset.count(sample);
            
            classifiers_decisions = classifiers.libsvm.x_do_classify(sample,obj.support_vectors_count,obj.support_vectors,obj.coeffs,obj.rhos,obj.kernel_code,obj.kernel_param1,obj.kernel_param2,obj.reg_param,logger.new_classifier('Classifying with each classifier'));
            
            logger.message('Determining most probable class.');

            if obj.saved_labels_count == 2
                classifiers_probs_t1 = 1 ./ (1 + 2.71314 .^ (-classifiers_decisions));
                classifiers_probs = [classifiers_probs_t1; 1 - classifiers_probs_t1];
                
                [max_probs,max_probs_idx] = max(classifiers_probs,[],1);
                
                labels_idx_hat = max_probs_idx;
                labels_confidence = bsxfun(@rdivide,classifiers_probs,max_probs);
            elseif tc.same(obj.multiclass_form,'1va')
                classifiers_probs = 1 ./ (1 + 2.71314 .^ (-classifiers_decisions));

                [max_probs,max_probs_idx] = max(classifiers_probs,[],1);

                labels_idx_hat = max_probs_idx;
                labels_confidence = bsxfun(@rdivide,classifiers_probs,max_probs);
            else
                pair_labels_idx = (classifiers_decisions < 0) + 1;
                partial_labels_idx = zeros(N,obj.classifiers_count);

                for ii = 1:obj.classifiers_count
                    partial_labels_idx(:,ii) = obj.saved_class_pair(ii,pair_labels_idx(ii,:))';
                end

                logger.message('Determining most probable class.');

                votes = hist(partial_labels_idx',obj.saved_labels_count);
                [max_votes,labels_idx_hat_t] = max(votes,[],1);

                labels_idx_hat = labels_idx_hat_t;
                labels_confidence = bsxfun(@rdivide,votes,max_votes);
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
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',1,log);
            
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
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 1);
            
            log.close();
            hnd.close();
            
            clearvars -except display;

            fprintf('    Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',1,log);
            
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
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 1);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Polynomial kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Polynomial',[2 3.5],1,'1va',1,log);
            
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
            assert(cl.kernel_code == 1);
            assert(cl.kernel_param1 == 2);
            assert(cl.kernel_param2 == 3.5);
            assert(tc.same(cl.kernel_type,'Polynomial'));
            assert(tc.same(cl.kernel_param,[2 3.5]));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 1);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Polynomial',[2 3.5],1,'1v1',1,log);
            
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
            assert(cl.kernel_code == 1);
            assert(cl.kernel_param1 == 2);
            assert(cl.kernel_param2 == 3.5);
            assert(tc.same(cl.kernel_type,'Polynomial'));
            assert(tc.same(cl.kernel_param,[2 3.5]));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 1);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Gaussian kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Gaussian',3.4,1,'1va',1,log);
            
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
            assert(cl.kernel_code == 2);
            assert(cl.kernel_param1 == 3.4);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Gaussian'));
            assert(tc.same(cl.kernel_param,3.4));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 1);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Gaussian',3.4,1,'1v1',1,log);
            
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
            assert(cl.kernel_code == 2);
            assert(cl.kernel_param1 == 3.4);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Gaussian'));
            assert(tc.same(cl.kernel_param,3.4));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 1);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Sigmoid kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Sigmoid',[0.05 0],1,'1va',1,log);
            
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
            assert(cl.kernel_code == 3);
            assert(cl.kernel_param1 == 0.05);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Sigmoid'));
            assert(tc.same(cl.kernel_param,[0.05 0]));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 1);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Sigmoid kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Sigmoid',[0.05 0],1,'1v1',1,log);
            
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
            assert(cl.kernel_code == 3);
            assert(cl.kernel_param1 == 0.05);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Sigmoid'));
            assert(tc.same(cl.kernel_param,[0.05 0]));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 1);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With multiple threads and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',3,log);
            
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
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With multiple threads and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_3();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',3,log);
            
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
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 3);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With two classes and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_2();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1va',1,log);
            
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
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1va'));
            assert(cl.num_threads == 1);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With two classes and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = utilstest.classifier_data_2();
            
            cl = classifiers.svm_kernel(s,ci,'Linear',0,1,'1v1',1,log);
            
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
            assert(cl.kernel_code == 0);
            assert(cl.kernel_param1 == 0);
            assert(cl.kernel_param2 == 0);
            assert(tc.same(cl.kernel_type,'Linear'));
            assert(tc.same(cl.kernel_param,0));
            assert(cl.reg_param == 1);
            assert(tc.same(cl.multiclass_form,'1v1'));
            assert(cl.num_threads == 1);
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "classify".\n');
            
            fprintf('    On clearly separated data.\n');
            
            fprintf('      Linear kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,log);
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
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,log);
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
            
            fprintf('      Polynomial kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,log);
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
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,log);
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
            
            fprintf('      Gaussian kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,log);
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
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,log);
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
            
            fprintf('      Sigmoid kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Sigmoid',[0.05 0],1,'1va',1,log);
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
            
            fprintf('      Sigmoid kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Sigmoid',[0.05 0],1,'1v1',1,log);
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
            
            fprintf('      Linear kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,log);
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
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,log);
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
            
            fprintf('      Polynomial kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,log);
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
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,log);
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
            
            fprintf('      Gaussian kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,log);
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
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,log);
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
            
            fprintf('      Sigmoid kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Sigmoid',[0.05 0],1,'1va',1,log);
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
            
            fprintf('      Sigmoid kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_clear_data_2();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Sigmoid',[0.05 0],1,'1v1',1,log);
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
            
            fprintf('      Linear kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,log);
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
            
            fprintf('      Linear kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,log);
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
            
            fprintf('      Polynomial kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,log);
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
            
            fprintf('      Polynomial kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,log);
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
            
            fprintf('      Gaussian kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,log);
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
            
            fprintf('      Gaussian kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,log);
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
            
            fprintf('      Sigmoid kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Sigmoid',[0.05 0],1,'1va',1,log);
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
            
            fprintf('      Sigmoid kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_mostly_clear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Sigmoid',[0.05 0],1,'1va',1,log);
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
            
            fprintf('      Linear kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1va',1,log);
            
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
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Linear',0,1,'1v1',1,log);
            
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
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1va',1,log);
            
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
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Polynomial',[2 3.5],1,'1v1',1,log);
            
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
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1va',1,log);
            
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
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Gaussian',3.4,1,'1v1',1,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Sigmoid kernel and One-vs-All multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Sigmoid',[0.05 0],1,'1va',1,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('      Sigmoid kernel and One-vs-One multiclass handling.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s_tr,s_ts,ci_tr,ci_ts] = utilstest.classifier_unclear_data_3();
            
            cl = classifiers.svm_kernel(s_tr,ci_tr,'Sigmoid',[0.05 0],1,'1v1',1,log);
            
            if exist('display','var') && (display == true)
                utilstest.show_classification_border(cl,s_tr,s_ts,ci_tr,ci_ts,[-1 5 -1 5]);
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
