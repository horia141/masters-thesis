function [best_param,param_list_extra] = crossvalidate_lin(full_sample,class_info,classifier_ctor_fn,param_list,fold_count,logger)
    assert(tc.dataset_record(full_sample));
    assert(tc.scalar(class_info));
    assert(tc.classification_info(class_info));
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
    assert(class_info.compatible(full_sample));
    
    [idx_tr,idx_cv] = class_info.partition('kfold',fold_count);
    
    best_param = struct('score_avg',-inf,'scores',-1,'times',-1);
    
    param_list_extra = param_list;
    [param_list_extra.scores] = deal(zeros(1,fold_count));
    [param_list_extra.score_avg] = deal(-inf);
    [param_list_extra.score_std] = deal(-inf);
    [param_list_extra.times] = deal(zeros(1,fold_count));
    [param_list_extra.time_avg] = deal(-inf);
    [param_list_extra.time_std] = deal(-inf);
    
    full_keeper = tic();
    
    for ii = 1:length(param_list)
        logger.beg_node('Configuration %d/%d',ii,length(param_list));
        
        logger.beg_node('Parameters');
        logger.message(params.to_string(param_list(ii)));
        logger.end_node();
        
        for fold = 1:fold_count
            logger.beg_node('Fold %d',fold);
            
            fold_keeper = tic();
            
            tr_sample = dataset.subsample(full_sample,idx_tr(fold,:));
            tr_sample_ci = class_info.subsample(idx_tr(fold,:));
            cv_sample = dataset.subsample(full_sample,idx_cv(fold,:));
            cv_sample_ci = class_info.subsample(idx_cv(fold,:));

            try
                cl = classifier_ctor_fn(tr_sample,tr_sample_ci,param_list(ii),logger.new_classifier('Training classifier on cross-validation set'));
                [~,~,param_list_extra(ii).scores(fold),~,~] = cl.classify(cv_sample,cv_sample_ci,logger.new_classifier('Classifying cross-validation set'));
            catch exp
                if strcmp(exp.identifier,'master:NoConvergence')
                    logger.message('Failed to converge with this configuration!');
                    param_list_extra(ii).scores(fold) = NaN;
                else
                    rethrow(exp);
                end
            end
                
            param_list_extra(ii).times(fold) = toc(fold_keeper);
                
            logger.message('Fold score: %.2f',param_list_extra(ii).scores(fold));
            logger.message('Fold time: %.2f',param_list_extra(ii).times(fold));
                
            logger.end_node();
        end
        
        param_list_extra(ii).score_avg = mean(param_list_extra(ii).scores(~isnan(param_list_extra(ii).scores)));
        param_list_extra(ii).score_std = std(param_list_extra(ii).scores(~isnan(param_list_extra(ii).scores)));
        param_list_extra(ii).time_avg = mean(param_list_extra(ii).times);
        param_list_extra(ii).time_std = std(param_list_extra(ii).times);
            
        logger.message('Average score: %.2f+/-%.2f',param_list_extra(ii).score_avg,param_list_extra(ii).score_std);
        logger.message('Total time: %.2fs',sum(param_list_extra(ii).times));
        logger.message('Average time: %.2f+/-%.2f',param_list_extra(ii).time_avg,param_list_extra(ii).time_std);
        
        logger.beg_node('Best parameters');
        p_best_param = rmfield(best_param,{'scores' 'times'});
        logger.message(params.to_string(p_best_param));

        if param_list_extra(ii).score_avg > best_param.score_avg
            logger.message('New best score!');
            
            best_param = param_list_extra(ii);
        end
        
        logger.end_node();
        
        logger.end_node();
    end
    
    total_sec_count = toc(full_keeper);
    
    logger.message('Total time: %.2fs',total_sec_count);
end
