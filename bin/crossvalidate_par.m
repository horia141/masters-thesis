function [best_param,param_list_extra] = crossvalidate_par(full_sample,class_info,classifier_ctor_fn,param_list,fold_count,logger)
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
    assert(fold_count == 10);
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
    
    job_id = randi(10000000);
    
    transfer_in_path = sprintf('transfer%d.in.mat',job_id);
    save(transfer_in_path,'-v7.3','full_sample','class_info','classifier_ctor_fn','param_list','idx_tr','idx_cv');
    
    full_keeper = tic();
    
    for ii = 1:length(param_list)
        logger.beg_node('Configuration %d/%d',ii,length(param_list));
        
        logger.beg_node('Parameters');
        logger.message(params.to_string(param_list(ii)));
        logger.end_node();
        
        transfer_out_path = 'transfer%d-config%d-fold%d.mat';
        
        [res(1),text] = system(sprintf('bash -c "ssh coman@sfb01.inb.uni-luebeck.de %s/crossvalidate_par_worker.sh %d %d 1 %s %s > /dev/null 2>&1 &"',pwd(),job_id,ii,transfer_in_path,transfer_out_path));
        [res(2),text] = system(sprintf('bash -c "ssh coman@sfb01.inb.uni-luebeck.de %s/crossvalidate_par_worker.sh %d %d 2 %s %s > /dev/null 2>&1 &"',pwd(),job_id,ii,transfer_in_path,transfer_out_path));
        [res(3),text] = system(sprintf('bash -c "ssh coman@sfb01.inb.uni-luebeck.de %s/crossvalidate_par_worker.sh %d %d 3 %s %s > /dev/null 2>&1 &"',pwd(),job_id,ii,transfer_in_path,transfer_out_path));
        [res(4),text] = system(sprintf('bash -c "ssh coman@sfb01.inb.uni-luebeck.de %s/crossvalidate_par_worker.sh %d %d 4 %s %s > /dev/null 2>&1 &"',pwd(),job_id,ii,transfer_in_path,transfer_out_path));
        [res(5),text] = system(sprintf('bash -c "ssh coman@sfb02.inb.uni-luebeck.de %s/crossvalidate_par_worker.sh %d %d 5 %s %s > /dev/null 2>&1 &"',pwd(),job_id,ii,transfer_in_path,transfer_out_path));
        [res(6),text] = system(sprintf('bash -c "ssh coman@sfb02.inb.uni-luebeck.de %s/crossvalidate_par_worker.sh %d %d 6 %s %s > /dev/null 2>&1 &"',pwd(),job_id,ii,transfer_in_path,transfer_out_path));
        [res(7),text] = system(sprintf('bash -c "ssh coman@sfb02.inb.uni-luebeck.de %s/crossvalidate_par_worker.sh %d %d 7 %s %s > /dev/null 2>&1 &"',pwd(),job_id,ii,transfer_in_path,transfer_out_path));
        [res(8),text] = system(sprintf('bash -c "ssh coman@sfb02.inb.uni-luebeck.de %s/crossvalidate_par_worker.sh %d %d 8 %s %s > /dev/null 2>&1 &"',pwd(),job_id,ii,transfer_in_path,transfer_out_path));
        [res(9),text] = system(sprintf('bash -c "ssh coman@grid30.inb.uni-luebeck.de %s/crossvalidate_par_worker.sh %d %d 9 %s %s > /dev/null 2>&1 &"',pwd(),job_id,ii,transfer_in_path,transfer_out_path));
        [res(10),text] = system(sprintf('bash -c "ssh coman@grid31.inb.uni-luebeck.de %s/crossvalidate_par_worker.sh %d %d 10 %s %s > /dev/null 2>&1 &"',pwd(),job_id,ii,transfer_in_path,transfer_out_path));

        if any(res ~= 0)
            throw(MException('master:SystemError','Could not start workers!'));
        end

        for fold = 1:fold_count
            fold_is_done = false;

            while ~fold_is_done
                if exist(sprintf(transfer_out_path,job_id,ii,fold),'file') == 2
                    fold_is_done = true;
                else
                    fold_is_done = false;
                    pause(1);
                end
            end
        end
        
        for fold = 1:fold_count
            logger.beg_node('Fold %d',fold);

            fold_res = load(sprintf(transfer_out_path,job_id,ii,fold));

            param_list_extra(ii).scores(fold) = fold_res.fold_score;
            param_list_extra(ii).times(fold) = fold_res.fold_time;
            
            logger.beg_classifier('Training and classification details:')
            logger.message(fold_res.log_output);
            logger.end_classifier();
            
            logger.message('Fold score: %.2f',param_list_extra(ii).scores(fold));
            logger.message('Fold time: %.2f',param_list_extra(ii).times(fold));
            
            logger.end_node();

            system(sprintf('rm %s',sprintf(transfer_out_path,job_id,ii,fold)));
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
    
    system(sprintf('rm %s',transfer_in_path));
    
    total_sec_count = toc(full_keeper);
    
    logger.message('Total time: %.2fs',total_sec_count);
end
