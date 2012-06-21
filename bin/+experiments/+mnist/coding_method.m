%% Setup experiment.

FOLD_COUNT = 10;
GOOD_DICT_PATH = '../explogs/mnist/good_dict_sparsenet_11x11.mat';
RESULTS_PATH = '../explogs/mnist/coding_method/results.mat';
TRAIN_WORKERS_COUNT = 45;
CLASSIFY_WORKERS_COUNT = 48;

good_dict_data = load(GOOD_DICT_PATH);

param_desc.dictionary_type = 'Dict';
param_desc.dictionary_params = {{good_dict_data.saved_dict11 'Corr' {}} ...
                                {good_dict_data.saved_dict11 'MP' 1} ...
                                {good_dict_data.saved_dict11 'MP' 2} ...
                                {good_dict_data.saved_dict11 'MP' 3} ...
                                {good_dict_data.saved_dict11 'MP' 4} ...
                                {good_dict_data.saved_dict11 'MP' 5} ...
                                {good_dict_data.saved_dict11 'MP' 7} ...
                                {good_dict_data.saved_dict11 'MP' 10} ...
                                {good_dict_data.saved_dict11 'MP' 15} ...
                                {good_dict_data.saved_dict11 'MP' 20} ...
                                {good_dict_data.saved_dict11 'MP' 50} ...
                                {good_dict_data.saved_dict11 'Euclidean' {}}};
param_desc.window_size = 11;
param_desc.window_step = 1;
param_desc.nonlinear_type = 'Logistic';
param_desc.nonlinear_params = {};
param_desc.reduce_type = 'Sqr';
param_desc.reduce_spread = 4;

param_list = params.gen_all(param_desc,...
                            @(p)mod(28 - 1,p.window_step) == 0,...
                            @(p)mod((28 - 1) / p.window_step + 1,p.reduce_spread) == 0);

param_desc_class.reg = logspace(-1.5,0.3,10);
param_list_class = params.gen_all(param_desc_class);

hnd = logging.handlers.stdout(logging.level.Transform);
logg = logging.logger({hnd});

%% Start experiment.

logg.beg_experiment('Experiment "MNIST - Coding Method"');

[m_tr,m_tr_ci,m_ts,m_ts_ci] = utilstest.load_mnist(logg.new_node('Loading MNIST dataset'));

alphas = cell(1,12);
t_wsrs = cell(1,12);
cls_best = cell(1,12);
scores = zeros(1,12);

logg.beg_node('Evaluating classifier performance for each configuration');

for ii = 1:length(param_list)
    logg.beg_node('Configuration %d/%d',ii,length(param_list));
    
    logg.beg_node('Parameters');
    p_param_list = rmfield(param_list(ii),{'dictionary_type' 'dictionary_params'});
    p_param_list.coding_method = param_list(ii).dictionary_params{2};
    p_param_list.coding_params = param_list(ii).dictionary_params{3};
    logg.message(params.to_string(p_param_list));
    logg.end_node();
    
   [alphas{ii},~,t_wsrs{ii},cls_best{ii},~,~,~,~,scores(ii),~,~] = ...
        experiments.mnist.run(m_tr,m_tr_ci,m_ts,m_ts_ci,...
                  @(s,l)transforms.image.window_sparse_recoder(s,10,param_list(ii).window_size,param_list(ii).window_size,0,...
                                                                 param_list(ii).dictionary_type,param_list(ii).dictionary_params,...
                                                                 param_list(ii).window_step,param_list(ii).nonlinear_type,param_list(ii).nonlinear_params,...
                                                                 param_list(ii).reduce_type,param_list(ii).reduce_spread,l),...
                  @(s,ci,p,l)classifiers.svm_linear(s,ci,'Primal','L2','L2',p.reg,'1v1',[TRAIN_WORKERS_COUNT CLASSIFY_WORKERS_COUNT],l),param_list_class,FOLD_COUNT,...
                  logg.new_node('Finding best classifier for current configuration'));

    logg.message('Saving intermediate results.');
    
    save(RESULTS_PATH,'-v7.3','alphas','t_wsrs','cls_best','scores','ii','param_desc','param_list');
    
    logg.end_node();
end

logg.end_node();

%% Stop experiment and save results.

logg.message('Saving final results.');
    
save(RESULTS_PATH,'-v7.3','alphas','t_wsrs','cls_best','scores','param_desc','param_list');

logg.message('Experiment done ... Terminating');

logg.end_experiment();

logg.close();
hnd.close();

%% Gather all results for plotting.

if exist('RESULTS_PATH','var')
    load(RESULTS_PATH);
else
    load ../explogs/mnist/coding_method/results.mat
end
