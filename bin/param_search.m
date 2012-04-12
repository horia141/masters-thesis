function [best_param,params_list] = param_search(params_desc,params_desc_filters,dataset_d,train_crossval_ratio,architecture_ctor_fn,logger)
    assert(tc.scalar(params_desc));
    assert(tc.struct(params_desc));
    % MORE COMPLETE CHECK HERE FOR PARAMS_DESC
    assert(tc.empty(params_desc_filters) || tc.vector(params_desc_filters));
    assert(tc.empty(params_desc_filters) || tc.cell(params_desc_filters));
    assert(tc.empty(params_desc_filters) || tc.checkf(@tc.scalar,params_desc_filters,true));
    assert(tc.empty(params_desc_filters) || tc.checkf(@tc.function_h,params_desc_filters,true));
    assert(tc.scalar(dataset_d));
    assert(tc.dataset(dataset_d));
    assert(dataset_d.samples_count >= 1);
    assert(tc.scalar(train_crossval_ratio));
    assert(tc.unitreal(train_crossval_ratio));
    assert(train_crossval_ratio > 0);
    assert(train_crossval_ratio < 1);
    assert(tc.scalar(architecture_ctor_fn));
    assert(tc.function_h(architecture_ctor_fn));
    assert(tc.scalar(logger));
    assert(tc.logging_logger(logger));
    assert(logger.active);
    
    logger.message('Generating all parameter combinations.');
    
    params_list = params.gen_all(params_desc,params_desc_filters{:});
    
    logger.message('Splitting full training data into training and crossvalidation sets.');
    
    [train_idx,crossval_idx] = dataset_d.partition('holdout',train_crossval_ratio);
    
    dataset_train = dataset_d.subsamples(train_idx);
    dataset_crossval = dataset_d.subsamples(crossval_idx);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                best_score = -inf;
    best_param = -1;
    
    for ii = 1:length(params_list)
        logger.beg_node('Parameter set #%d/%d',ii,length(params_list));
        logger.beg_node('Configuration');
        logger.message(params.to_string(params_list(ii)));
        logger.end_node();
        
        try
            ar = architecture_ctor_fn(dataset_train,params_list(ii),logger.new_architecture('Training architecture'));
            [~,~,~,~,params_list(ii).score,~,~] = ar.classify(dataset_crossval,logger.new_architecture('Evaluating performance on crossvalidation set'));
            
            logger.message('Score: %.2f',params_list(ii).score);
        
            if params_list(ii).score > best_score
                logger.message('We have a new best parameter set!');
            
                best_score = params_list(ii).score;
                best_param = params_list(ii);
            end
        catch exp
            if strcmp(exp.identifier,'master:NoConvergence')
                logger.message('The classifier could not be made to converge for this configuration!');
                logger.message('Reason: %s',exp.message);
                logger.message('Skipping configuration!');
            else
                rethrow(exp);
            end
        end        
        
        logger.end_node();
    end
    
    if tc.number(best_param) && (best_param == -1)
        throw(MException('master:NoGoodConfiguration',...
                         'Function "param_search" was unable to find a valid configuration in cross-validation search!'));
    end
end
