classdef arch_alpha
    properties (GetAccess=public,SetAccess=immutable)
        image_patch_row_count;
        image_patch_col_count;
        pooled_patch_row_count;
        pooled_patch_col_count;
        input_geometry;
        t_dc_offset;
        t_mean_substract;
        t_zca;
        t_sparse;
        classifier;
        patches_count;
        patch_row_count;
        patch_col_count;
        patch_required_variance;
        coding_fn;
        word_count;
        coeffs_count;
        initial_learning_rate;
        final_learning_rate;
        max_iter_count;
        window_step;
        reduce_fn;
        reduce_spread;
    end
    
    methods (Access=public)
        function [obj] = arch_alpha(train_sample_plain,class_info,patches_count,patch_row_count,patch_col_count,patch_required_variance,coding_fn,word_count,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count,window_step,reduce_fn,reduce_spread,classifier_ctor_fn,classifier_params,log)
            assert(tc.dataset_image(train_sample_plain));
            assert(size(train_sample_plain,3) == 1);
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info));
            assert(tc.scalar(patches_count));
            assert(tc.scalar(patches_count));
            assert(tc.natural(patches_count));
            assert(patches_count >= 1);
            assert(tc.scalar(patch_row_count));
            assert(tc.natural(patch_row_count));
            assert(patch_row_count >= 1);
            assert(tc.scalar(patch_col_count));
            assert(tc.natural(patch_col_count));
            assert(patch_col_count >= 1);
            assert(tc.scalar(patch_required_variance));
            assert(tc.number(patch_required_variance));
            assert(patch_required_variance >= 0);
            assert(tc.scalar(coding_fn));
            assert(tc.function_h(coding_fn));
            assert(tc.one_of(coding_fn,@transforms.sparse.gdmp2.correlation,@transforms.sparse.gdmp2.matching_pursuit,@transforms.sparse.gdmp2.matching_pursuit_alpha,@transforms.sparse.gdmp2.ortho_matching_pursuit));
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
            assert(tc.scalar(window_step));
            assert(tc.natural(window_step));
            assert(window_step >= 1);
            assert(tc.scalar(reduce_fn));
            assert(tc.function_h(reduce_fn));
            assert(tc.one_of(reduce_fn,@architectures.arch_alpha.sqr,@architectures.arch_alpha.max));
            assert(tc.scalar(reduce_spread));
            assert(tc.natural(reduce_spread));
            assert(reduce_spread >= 1);
            assert(tc.scalar(classifier_ctor_fn));
            assert(tc.function_h(classifier_ctor_fn));
            assert(tc.empty(classifier_params) || tc.vector(classifier_params));
            assert(tc.empty(classifier_params) || tc.cell(classifier_params));
            assert(tc.empty(classifier_params) || tc.checkf(@tc.value,classifier_params));
            assert(tc.scalar(log));
            assert(tc.logging_logger(log));
            assert(log.active);
            assert(mod(size(train_sample_plain,1)- patch_row_count,window_step) == 0); % A BIT OF A HACK
            assert(mod(size(train_sample_plain,2)- patch_col_count,window_step) == 0);
            assert(mod((size(train_sample_plain,1) - patch_row_count) / window_step + 1,reduce_spread) == 0);
            assert(mod((size(train_sample_plain,2) - patch_col_count) / window_step + 1,reduce_spread) == 0);
            assert(class_info.compatible(train_sample_plain));
            
            N = dataset.count(train_sample_plain);
            [d,dr,dc,~] = dataset.geometry(train_sample_plain);
            
            image_patch_row_count_t = (dr - patch_row_count) / window_step + 1;
            image_patch_col_count_t = (dc - patch_col_count) / window_step + 1;
            pooled_image_patch_row_count = image_patch_row_count_t / reduce_spread;
            pooled_image_patch_col_count = image_patch_col_count_t / reduce_spread;
            
            t_patches = transforms.image.patch_extract(train_sample_plain,patches_count,patch_row_count,patch_col_count,patch_required_variance,log.new_transform('Building patch extract transform'));
            patches_1 = t_patches.code(train_sample_plain,log.new_transform('Extracting patches'));
            patches_2 = dataset.flatten_image(patches_1);
            t_dc_offset_t = transforms.record.dc_offset(patches_2,log.new_transform('Building DC offset removal transform'));
            patches_3 = t_dc_offset_t.code(patches_2,log.new_transform('Removing DC component'));
            t_mean_substract_t = transforms.record.mean_substract(patches_3,log.new_transform('Building mean substract transform'));
            patches_4 = t_mean_substract_t.code(patches_3,log.new_transform('Substracting mean'));
            t_zca_t = transforms.record.zca(patches_4,log.new_transform('Building ZCA transform'));
            patches_5 = t_zca_t.code(patches_4,log.new_transform('Applying ZCA transform'));
            
            t_sparse_t = transforms.sparse.gdmp2(patches_5,coding_fn,word_count,coeffs_count,initial_learning_rate,final_learning_rate,max_iter_count,log.new_transform('Building sparse coder for patches'));

            clear patches_1 patches_2 patches_3 patches_4 patches_5;
            
            log.beg_node('Window coding dataset');

            patch_coded_data = zeros(N,2 * word_count,image_patch_row_count_t,image_patch_col_count_t);
            ii_image = 1;
            jj_image = 1;
            
            for ii = 1:window_step:(dr - patch_row_count + 1)
                for jj = 1:window_step:(dr - patch_col_count + 1)
                    log.beg_node('Window %02d%02d',ii_image,jj_image);

                    temp_train_image_1 = train_sample_plain(ii:(ii + patch_row_count - 1),jj:(jj + patch_col_count - 1),:,:);
                    temp_train_image_2 = dataset.flatten_image(temp_train_image_1);
                    temp_train_image_3 = t_dc_offset_t.code(temp_train_image_2,log.new_transform('Removing DC component'));
                    temp_train_image_4 = t_mean_substract_t.code(temp_train_image_3,log.new_transform('Substracting mean'));
                    temp_train_image_5 = t_zca_t.code(temp_train_image_4,log.new_transform('Applying ZCA transform'));
                    temp_train_image_coded = t_sparse_t.code(temp_train_image_5,log.new_transform('Computing sparse representation'));
                    patch_coded_data(:,:,ii_image,jj_image) = temp_train_image_coded;
                    clear temp_train_image_1 temp_train_image_2 temp_train_image_3 temp_traim_image_4 temp_train_image_5 temp_train_image_coded;
                    jj_image = jj_image + 1;

                    log.end_node();
                end
                
                ii_image = ii_image + 1;
                jj_image = 1;
            end
            
            log.end_node();
            
            log.beg_node('Pooling coded image patches');
            
            index_size_tmp = pooled_image_patch_row_count * pooled_image_patch_col_count;
            new_samples = zeros(N,2 * word_count * index_size_tmp);
            log_batch_size = ceil(2 * word_count / 25);
            local = zeros(N,index_size_tmp);
            
            for word = 1:2 * word_count
                if mod(word - 1,log_batch_size) == 0
                    log.message('Pooling for features %d to %d',word,min(word + log_batch_size - 1,2 * word_count));
                end
                
                ii_image = 1;
                jj_image = 1;
                
                for ii = 1:reduce_spread:image_patch_row_count_t
                    for jj = 1:reduce_spread:image_patch_col_count_t
                        local(:,(ii_image - 1) * pooled_image_patch_col_count + jj_image) = reduce_fn(patch_coded_data(:,word,(ii:(ii + reduce_spread - 1)),jj:(jj + reduce_spread - 1)));
                        jj_image = jj_image + 1;
                    end
                    
                    ii_image = ii_image + 1;
                    jj_image = 1;
                end
                
                new_samples(:,((word - 1) * index_size_tmp+1):(word * index_size_tmp)) = local;
            end
            
            log.end_node();
            
            classifier_t = classifier_ctor_fn(new_samples,class_info,classifier_params{:},log.new_classifier('Training classifier'));

            obj.image_patch_row_count = image_patch_row_count_t;
            obj.image_patch_col_count = image_patch_col_count_t;
            obj.pooled_patch_row_count = pooled_image_patch_row_count;
            obj.pooled_patch_col_count = pooled_image_patch_col_count;
            obj.input_geometry = [d,dc,dr,1];
            obj.t_dc_offset = t_dc_offset_t;
            obj.t_mean_substract = t_mean_substract_t;
            obj.t_zca = t_zca_t;
            obj.t_sparse = t_sparse_t;
            obj.classifier = classifier_t;
            obj.patches_count = patches_count;
            obj.patch_row_count = patch_row_count;
            obj.patch_col_count = patch_col_count;
            obj.patch_required_variance = patch_required_variance;
            obj.coding_fn = coding_fn;
            obj.word_count = word_count;
            obj.coeffs_count = coeffs_count;
            obj.initial_learning_rate = initial_learning_rate;
            obj.final_learning_rate = final_learning_rate;
            obj.max_iter_count = max_iter_count;
            obj.window_step = window_step;
            obj.reduce_fn = reduce_fn;
            obj.reduce_spread = reduce_spread;
        end
        
        function [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = classify(obj,sample_plain,class_info,log)
            assert(tc.scalar(obj));
            assert(isa(obj,'architectures.arch_alpha'));
            assert(tc.dataset_image(sample_plain));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info) || (class_info == -1));
            assert(tc.scalar(log));
            assert(tc.logging_logger(log));
            assert(log.active);
            assert(dataset.geom_compatible(obj.input_geometry,dataset.geometry(sample_plain)));
            assert(~tc.classification_info(class_info) || (tc.same(obj.classifier.saved_labels,class_info.labels)));
            assert(~tc.classification_info(class_info) || class_info.compatible(sample_plain));
            
            log.beg_node('Window coding dataset');
            
            N = dataset.count(sample_plain);
            [~,dr,dc,~] = dataset.geometry(sample_plain);
            
            patch_coded_data = zeros(N,2 * obj.word_count,obj.image_patch_row_count,obj.image_patch_col_count);
            ii_image = 1;
            jj_image = 1;
            
            for ii = 1:obj.window_step:(dr - obj.patch_row_count + 1)
                for jj = 1:obj.window_step:(dc - obj.patch_col_count + 1)
                    log.beg_node('Window %02d%02d',ii_image,jj_image);
                    
                    temp_train_image_1 = sample_plain(ii:(ii + obj.patch_row_count - 1),jj:(jj + obj.patch_col_count - 1),:,:);
                    temp_train_image_2 = dataset.flatten_image(temp_train_image_1);
                    temp_train_image_3 = obj.t_dc_offset.code(temp_train_image_2,log.new_transform('Removing DC component'));
                    temp_train_image_4 = obj.t_mean_substract.code(temp_train_image_3,log.new_transform('Substracting mean'));
                    temp_train_image_5 = obj.t_zca.code(temp_train_image_4,log.new_transform('Applying ZCA transform'));
                    temp_train_image_coded = obj.t_sparse.code(temp_train_image_5,log.new_transform('Computing sparse representation'));
                    patch_coded_data(:,:,ii_image,jj_image) = temp_train_image_coded;
                    clear temp_train_image_1 temp_train_image_2 temp_train_image_3 temp_traim_image_4 temp_traim_image_5 temp_train_image_coded;
                    jj_image = jj_image + 1;
                    
                    log.end_node();
                end
                
                ii_image = ii_image + 1;
                jj_image = 1;
            end
            
            log.end_node();
            
            log.beg_node('Pooling coded image patches');
            
            index_size_tmp = obj.pooled_patch_row_count * obj.pooled_patch_col_count;
            new_samples = zeros(N,2 * obj.word_count * index_size_tmp);
            log_batch_size = ceil(2 * obj.word_count / 25);
            local = zeros(N,index_size_tmp);
            
            for word = 1:2 * obj.word_count
                if mod(word - 1,log_batch_size) == 0
                    log.message('Pooling for features %d to %d',word,min(word + log_batch_size - 1,2 * obj.word_count));
                end
                
                ii_image = 1;
                jj_image = 1;
                
                for ii = 1:obj.reduce_spread:obj.image_patch_row_count
                    for jj = 1:obj.reduce_spread:obj.image_patch_col_count
                        local(:,(ii_image - 1) * obj.pooled_patch_col_count + jj_image) = obj.reduce_fn(patch_coded_data(:,word,(ii:(ii + obj.reduce_spread - 1)),jj:(jj + obj.reduce_spread - 1)));
                        jj_image = jj_image + 1;
                    end
                    
                    ii_image = ii_image + 1;
                    jj_image = 1;
                end
                
                new_samples(:,((word - 1) * index_size_tmp+1):(word * index_size_tmp)) = local;
            end
            
            log.end_node();
            

            [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = obj.classifier.classify(new_samples,class_info,log.new_classifier('Classifying'));
        end
    end
    
    methods (Static,Access=public)
        function [o] = sqr(A)
            o = sum(sum(A .^ 2,3),4);
        end
        
        function [o] = max(A)
            o = max(max(A,[],3),[],4);
        end
    end
end
