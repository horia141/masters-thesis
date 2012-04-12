function [] = small_images(title,...
                           mnist_full_images_path,mnist_full_labels_path,mnist_test_images_path,mnist_test_labels_path,...
                           new_digit_row_count,new_digit_col_count,...
                           params_desc,params_desc_filters,train_crossval_ratio,...
                           architecture_ctor_fn,...
                           details_logfile_path,results_email_addrs,error_email_addrs,results_error_sender)
    assert(tc.scalar(title));
    assert(tc.string(title));
    assert(tc.scalar(mnist_full_images_path));
    assert(tc.string(mnist_full_images_path));
    assert(tc.scalar(mnist_full_labels_path));
    assert(tc.string(mnist_full_labels_path));
    assert(tc.scalar(mnist_test_images_path));
    assert(tc.string(mnist_test_images_path));
    assert(tc.scalar(mnist_test_labels_path));
    assert(tc.string(mnist_test_labels_path));
    assert(tc.scalar(new_digit_row_count));
    assert(tc.natural(new_digit_row_count));
    assert(new_digit_row_count >= 1);
    assert(tc.scalar(new_digit_col_count));
    assert(tc.natural(new_digit_col_count));
    assert(new_digit_col_count >= 1);
    assert(tc.scalar(params_desc));
    assert(tc.struct(params_desc));
    % MORE COMPLETE CHECK HERE FOR PARAMS_DESC
    assert(tc.empty(params_desc_filters) || tc.vector(params_desc_filters));
    assert(tc.empty(params_desc_filters) || tc.cell(params_desc_filters));
    assert(tc.empty(params_desc_filters) || tc.checkf(@tc.scalar,params_desc_filters,true));
    assert(tc.empty(params_desc_filters) || tc.checkf(@tc.function_h,params_desc_filters,true));
    assert(tc.scalar(train_crossval_ratio));
    assert(tc.unitreal(train_crossval_ratio));
    assert(train_crossval_ratio > 0);
    assert(train_crossval_ratio < 1);
    assert(tc.scalar(architecture_ctor_fn));
    assert(tc.function_h(architecture_ctor_fn));
    assert(tc.scalar(details_logfile_path));
    assert(tc.string(details_logfile_path));
    assert(tc.vector(results_email_addrs));
    assert(tc.cell(results_email_addrs));
    assert(tc.checkf(@tc.scalar,results_email_addrs));
    assert(tc.checkf(@tc.string,results_email_addrs));
    assert(tc.vector(error_email_addrs));
    assert(tc.cell(error_email_addrs));
    assert(tc.checkf(@tc.scalar,error_email_addrs));
    assert(tc.checkf(@tc.string,error_email_addrs));
    assert(tc.scalar(results_error_sender));
    assert(tc.string(results_error_sender));
    
    hnd_overview = logging.handlers.stdout(logging.level.All);
    hnd_details = logging.handlers.file(details_logfile_path,logging.level.All);
    hnd_results = logging.handlers.sendmail(title,results_email_addrs,results_error_sender,'pc07.inb.uni-luebeck.de',logging.level.Results);
    hnd_error = logging.handlers.sendmail(title,error_email_addrs,results_error_sender,'pc07.inb.uni-luebeck.de',logging.level.Error);
    logger = logging.logger({hnd_overview,hnd_details,hnd_results,hnd_error});

    logger.beg_experiment(title);
    
    logger.beg_node('Loading MNIST images');
 
    mnist_full_raw = datasets.image.load_mnist(mnist_full_images_path,mnist_full_labels_path,logger.new_dataset_io('Full training data'));
    mnist_test_raw = datasets.image.load_mnist(mnist_test_images_path,mnist_test_labels_path,logger.new_dataset_io('Test data'));
    
    assert(mnist_full_raw.compatible(mnist_test_raw)); % HACK DI HACK. Should throw an error here

    logger.end_node();
    
    if new_digit_row_count ~= mnist_full_raw.row_count || ...
       new_digit_col_count ~= mnist_full_raw.col_count
        logger.beg_node('Resizing MNIST images');
 
        t_resize = transforms.image.resize(mnist_full_raw,new_digit_row_count,new_digit_col_count,logger.new_transform('Training resize transform'));

        mnist_full = t_resize.code(mnist_full_raw,logger.new_transform('Full training data'));
        mnist_test = t_resize.code(mnist_test_raw,logger.new_transform('Test data'));
 
        logger.end_node();
    
        clear mnist_full_raw;
        clear mnist_test_raw;
    else
        mnist_full = mnist_full_raw;
        mnist_test = mnist_test_raw;
    end
    
    logger.beg_node('Performing meta-search for best parameters');

    [best_params,params_list] = param_search(params_desc,params_desc_filters,mnist_full,train_crossval_ratio,architecture_ctor_fn,logger);

    logger.end_node();

    logger.beg_node('Best configuration [score = %.2f]',best_params.score);
    logger.message(params.to_string(best_params));
    logger.end_node();
    
    logger.beg_node('Evaluating performance on test set');

    ar = architecture_ctor_fn(mnist_full,best_params,logger.new_architecture('Training best architecture'));
    [~,~,~,~,score,confusion_matrix,~] = ar.classify(mnist_test,logger.new_architecture('Performing classification'));

    logger.end_node();

    logger.beg_results('Results');

    logger.message('Score: %.2f',score);
    logger.message('Confusion_matrix:');
    logger.message(utils.matrix_to_string(confusion_matrix,'%4d '));
    
    logger.beg_node('All experiments results');
    logger.message(params.to_table_string(params_list));
    logger.end_node();

    logger.end_results();
    
    logger.end_experiment();

    logger.close();
    hnd_overview.close();
    hnd_details.close();
    hnd_results.close();
    hnd_error.close();
end
