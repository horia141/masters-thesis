function [alpha,beta,t_wsr,cl_best,m_tr_coded,m_ts_coded,labels_hat,labels_confidence,score,conf_mat,misclassified] = ...
            run(m_tr,m_tr_ci,m_ts,m_ts_ci,window_sparse_coder_ctor_fn,classifier_ctor_fn,param_list,fold_count,logger)
    assert(tc.dataset_image(m_tr));
    assert(tc.scalar(m_tr_ci));
    assert(tc.classification_info(m_tr_ci));
    assert(tc.dataset_image(m_ts));
    assert(tc.scalar(m_ts_ci));
    assert(tc.classification_info(m_ts_ci));
    assert(tc.scalar(window_sparse_coder_ctor_fn));
    assert(tc.function_h(window_sparse_coder_ctor_fn));
    assert(tc.scalar(classifier_ctor_fn));
    assert(tc.function_h(classifier_ctor_fn));
    assert(tc.vector(param_list));
    assert(tc.struct(param_list));
    assert(tc.scalar(fold_count));
    assert(tc.natural(fold_count));
    assert(fold_count >= 1);
    assert(tc.scalar(logger));
    assert(tc.logging_logger(logger));
    assert(logger.active);
    assert(m_tr_ci.compatible(m_tr));
    assert(m_ts_ci.compatible(m_ts));
    assert(dataset.geom_compatible(dataset.geometry(m_tr),dataset.geometry(m_ts)));
    
    tic();
    
    t_wsr = window_sparse_coder_ctor_fn(m_tr,logger.new_transform('Building window coder transform'));
    
    logger.message('New sample geometry: %d',t_wsr.output_geometry);
    
    coder_build_time = toc();
    
%     utilsdisplay.sparse_basis(t_wsr.t_dictionary.dict,sqrt(size(t_wsr.t_dictionary.dict,2)),sqrt(size(t_wsr.t_dictionary.dict,2)));
%     pause(5);
%     
    tic();
    
    m_tr_coded = t_wsr.code(m_tr,logger.new_transform('Coding training dataset'));
    m_ts_coded = t_wsr.code(m_ts,logger.new_transform('Coding test dataset'));
    
    coding_time = toc();
    
    tic();
    
    [alpha,beta] = crossvalidate(m_tr_coded,m_tr_ci,classifier_ctor_fn,param_list,fold_count,logger.new_node('Running a cross-validation search for best classifier parameters'));
    
    crossvalidation_time = toc();
    
    tic();
    
    cl_best = classifier_ctor_fn(m_tr_coded,m_tr_ci,alpha,logger.new_classifier('Training classifier with best parameters on whole training dataset'));
    [labels_hat,labels_confidence,score,conf_mat,misclassified] = cl_best.classify(m_ts_coded,m_ts_ci,logger.new_classifier('Evaluating performance on test dataset'));
    
    training_time = toc();
    
    logger.message('Window coder build time: %.0fs',coder_build_time);
    logger.message('Total coding time: %.0fs',coding_time);
    logger.message('Average time per instance: %.2fs',coding_time / (dataset.count(m_tr) + dataset.count(m_ts)));
    logger.message('Total crossvalidation time: %.0fs',crossvalidation_time);
    logger.message('Average time per classifier: %.0fs',crossvalidation_time / length(param_list));
    logger.message('Total training time: %.0fs',training_time);
    logger.message('Final score: %.4f',score);
end
