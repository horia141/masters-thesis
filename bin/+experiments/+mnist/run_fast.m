function [] = run_fast(results_out_path,patches_count,patch_row_count,patch_col_count,dictionary_type,dictionary_params,...
                                        window_step,nonlinear_type,nonlinear_params,reduce_type,reduce_spread)
    assert(tc.scalar(results_out_path));
    assert(tc.string(results_out_path));
    assert(tc.scalar(patches_count));
    assert(tc.scalar(patches_count));
    assert(tc.natural(patches_count));
    assert(patches_count >= 1);
    assert(tc.scalar(patch_row_count));
    assert(tc.natural(patch_row_count));
    assert(patch_row_count >= 1);
    assert(mod(patch_row_count,2) == 1);
    assert(tc.scalar(patch_col_count));
    assert(tc.natural(patch_col_count));
    assert(patch_col_count >= 1);
    assert(mod(patch_col_count,2) == 1);
    assert(tc.scalar(dictionary_type));
    assert(tc.string(dictionary_type));
    assert(tc.one_of(dictionary_type,'Dict','Learn:Grad','Learn:KMeans','Learn:NeuralGas','Random:Filters','Random:Instances'));
    assert(tc.empty(dictionary_params) || tc.vector(dictionary_params));
    assert(tc.empty(dictionary_params) || tc.cell(dictionary_params));
    assert(tc.scalar(window_step));
    assert(tc.natural(window_step));
    assert(window_step >= 1);
    assert(tc.scalar(nonlinear_type));
    assert(tc.string(nonlinear_type));
    assert(tc.one_of(nonlinear_type,'Linear','Logistic'));
    assert(tc.empty(nonlinear_params) || tc.vector(nonlinear_params));
    assert(tc.empty(nonlinear_params) || tc.cell(nonlinear_params));
    assert(tc.scalar(reduce_type));
    assert(tc.string(reduce_type));
    assert(tc.one_of(reduce_type,'Subsample','Sqr','Max','MinMax'));
    assert(tc.scalar(reduce_spread));
    assert(tc.natural(reduce_spread));
    assert(mod(28 - 1,window_step) == 0); % A BIT OF A HACK
    assert(mod(28 - 1,window_step) == 0); % A BIT OF A HACK
    assert(mod((28 - 1) / window_step + 1,reduce_spread) == 0);
    assert(mod((28 - 1) / window_step + 1,reduce_spread) == 0);
    
    hnd = logging.handlers.stdout(logging.level.Transform);
    logg = logging.logger({hnd});
    
    [m_tr,m_tr_ci,m_ts,m_ts_ci] = utilstest.load_mnist(logg);
    
    param_desc.reg = logspace(-1.5,0.7,10);
    param_list = params.gen_all(param_desc);
    
    [alpha,beta,t_wsr,cl_best,~,~,labels_hat,labels_confidence,score,conf_mat,misclassified] = ...
        experiments.mnist.run(m_tr,m_tr_ci,m_ts,m_ts_ci,...
                              @(s,l)transforms.image.window_sparse_recoder(s,patches_count,patch_row_count,patch_col_count,0.01,dictionary_type,dictionary_params,...
                                                                           window_step,nonlinear_type,nonlinear_params,reduce_type,reduce_spread,l),...
                              @(s,ci,p,l)classifiers.svm_linear(s,ci,'Primal','L2','L2',p.reg,'1v1',[45 48],l),...
                              param_list,10,logg);
                          
    save(results_out_path,'alpha','beta','t_wsr','cl_best','labels_hat','labels_confidence','score','conf_mat','misclassified');
end
