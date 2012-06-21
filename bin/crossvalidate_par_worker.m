function [] = crossvalidate_par_worker()
    job_id = sscanf(getenv('JOB_ID'),'%d');
    configuration = sscanf(getenv('CONFIGURATION'),'%d');
    fold = sscanf(getenv('TASK_ID'),'%d');
    transfer_in_path = getenv('TRANSFER_IN_PATH');
    transfer_out_path = getenv('TRANSFER_OUT_PATH');
    
    transfered_data = load(transfer_in_path);
    
    full_sample = transfered_data.full_sample;
    class_info = transfered_data.class_info;
    classifier_ctor_fn = transfered_data.classifier_ctor_fn;
    param_list = transfered_data.param_list;
    idx_tr = transfered_data.idx_tr;
    idx_cv = transfered_data.idx_cv;
    
    hnd = logging.handlers.testing(logging.level.All);
    logg = logging.logger({hnd});
    
    try
        fold_keeper = tic();
        
        tr_sample = dataset.subsample(full_sample,idx_tr(fold,:));
        tr_sample_ci = class_info.subsample(idx_tr(fold,:));
        cv_sample = dataset.subsample(full_sample,idx_cv(fold,:));
        cv_sample_ci = class_info.subsample(idx_cv(fold,:));
        
        cl = classifier_ctor_fn(tr_sample,tr_sample_ci,param_list(configuration),logg.new_classifier('Training classifier on cross-validation set'));
        [~,~,fold_score,~,~] = cl.classify(cv_sample,cv_sample_ci,logg.new_classifier('Classifying cross-validation set'));
                
        fold_time = toc(fold_keeper);
    catch exp
        if strcmp(exp.identifier,'master:NoConvergence')
            fold_score = -inf;
            fold_time = toc(fold_keeper);
        else
            % Silently fail here. Nothing much we can do if we don't want
            % an overly complex failure mechanism.
        end
    end
    
    logg.close();
    hnd.close();
    
    log_output = hnd.logged_data;

    save(sprintf(transfer_out_path,job_id,configuration,fold),'-v7.3','fold_score','fold_time','log_output');
    
    exit();
end
