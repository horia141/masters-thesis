function [best_param,param_list_extra] = crossvalidate(full_sample,class_info,classifier_ctor_fn,param_list,cv_tr_ratio,logger)
    assert(tc.dataset_record(full_sample));
    assert(tc.scalar(class_info));
    assert(tc.classification_info(class_info));
    assert(tc.scalar(classifier_ctor_fn));
    assert(tc.function_h(classifier_ctor_fn));
    assert(tc.vector(param_list));
    assert(tc.struct(param_list));
    assert(tc.scalar(cv_tr_ratio));
    assert(tc.unitreal(cv_tr_ratio));
    assert(cv_tr_ratio > 0);
    assert(cv_tr_ratio < 1);
    assert(tc.scalar(logger));
    assert(tc.logging_logger(logger));
    assert(logger.active);
    assert(class_info.compatible(full_sample));
    
    logger.message('Splitting full sample into test and cross-validation ones.');
    
    [idx_tr,idx_cv] = class_info.partition('holdout',cv_tr_ratio);
    
    tr_sample = dataset.subsample(full_sample,idx_tr);
    tr_sample_ci = class_info.subsample(idx_tr);
    cv_sample = dataset.subsample(full_sample,idx_cv);
    cv_sample_ci = class_info.subsample(idx_cv);
    
    logger.beg_node('Starting meta-search for best C');

    best_param = struct('score',-inf);
    
    param_list_extra = param_list;
    [param_list_extra.score] = deal(-inf);
    [param_list_extra(:).time] = deal(0);
    
    full_keeper = tic();
    
    for ii = 1:length(param_list)
        logger.beg_node('Configuration %d/%d',ii,length(param_list));
        
        logger.beg_node('Parameters');
        logger.message(params.to_string(param_list(ii)));
        logger.end_node();
        
        iter_keeper = tic();
        
        try
            cl = classifier_ctor_fn(tr_sample,tr_sample_ci,param_list(ii),logger.new_classifier('Training classifier on cross-validation set'));
            [~,~,param_list_extra(ii).score,~,~] = cl.classify(cv_sample,cv_sample_ci,logger.new_classifier('Classifying cross-validation set'));
            
            logger.message('Score: %.2f',param_list_extra(ii).score);
        catch exp
            if strcmp(exp.identifier,'master:NoConvergence')
                logger.message('Failed to converge with this configuration!');
                param_list_extra(ii).score = NaN;
            else
                rethrow(exp);
            end
        end
        
        param_list_extra(ii).time = toc(iter_keeper);
        
        logger.beg_node('Best parameters');
        logger.message(params.to_string(best_param));
        logger.end_node();
        logger.message('Time: %.2fs',param_list_extra(ii).time);

        if param_list_extra(ii).score > best_param.score
            logger.message('New best score!');
            
            best_param = param_list_extra(ii);
        end
        
        logger.end_node();
    end
    
    total_sec_count = toc(full_keeper);
    
    logger.message('Total time: %.2fs',total_sec_count);
    
    logger.end_node();
end
