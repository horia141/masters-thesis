%% Setup experiment-wide constants.

MODEL_SELECTION_RATIO = 'full';
CODER_REP_COUNT = 5;
CLASSIFIER_REG = 0.0025;
RESULTS_PATH = '../explogs/mnist/correlation_order/best1_coder_transforms/results_1.mat';

TRAIN_WORKER_COUNT = 45;
CLASSIFY_WORKER_COUNT = 48;

%% Build the list of coder configurations to test.

% Coding method.
param_desc_coder.patches_count = 10;
param_desc_coder.do_patch_zca = false;
param_desc_coder.dictionary_type = 'Random:Filters';
param_desc_coder.dictionary_params = {{512 'CorrOrder' [0.05 0.01 48]}};
% Coder transforms.
param_desc_coder.do_polarity_split = {false true};
param_desc_coder.nonlinear_type = {'Linear' 'Logistic'};
param_desc_coder.nonlinear_params = {};
param_desc_coder.reduce_type = {'Subsample' 'Sqr' 'Max' 'MaxMagnitude'};
% Coder geometry.
param_desc_coder.window_size = 9;
param_desc_coder.window_step = 1;
param_desc_coder.reduce_spread = 4;

param_list_coder = utils.params.gen_all(param_desc_coder,...
                                        @(p)mod(28 - 1,p.window_step) == 0,...
                                        @(p)mod((28 - 1) / p.window_step + 1,p.reduce_spread) == 0);
                        
%% Initialize the logging subsystem.

hnd = logging.handlers.stdout(logging.level.Experiment);
logg = logging.logger({hnd});

logg.beg_node('Experiment "MNIST - Correlation Order - Best Coder Transforms"');

%% Make sure we can write to the results file.

logg.message('Checking results file.');

if exist(RESULTS_PATH,'file')
    logg.end_node();
    logg.close();
    hnd.close();
    error('The results file "%s" already exists!',RESULTS_PATH);
else
    [results_dir,~,~] = fileparts(RESULTS_PATH);
    
    if ~exist(results_dir,'dir')
        logg.message('Building results file directory.');
        system(sprintf('mkdir -p "%s"',results_dir));
    end
end

%% Load the experiment data, separating it into the part used for model selection and the part used for final testing.

[d_tr,d_tr_ci,d_ts,d_ts_ci] = utils.load_dataset.mnist(logg.new_node('Loading MNIST dataset'));

%% Filter the experiment model selection data, retaining only a subsample from it.

logg.message('Extracting the data for model selection.');

if check.same(MODEL_SELECTION_RATIO,'full')
    N_tr = dataset.count(d_tr);
    coder_useful_idx = 1:N_tr;
    d_coder_useful = d_tr;
    d_coder_useful_ci = d_tr_ci;
else
    N_tr = dataset.count(d_tr);
    coder_useful_idx = randi(N_tr,1,min(N_tr,ceil(N_tr * MODEL_SELECTION_RATIO)));
    d_coder_useful = dataset.subsample(d_tr,coder_useful_idx);
    d_coder_useful_ci = d_tr_ci.subsample(coder_useful_idx);
end

%% For each coder configuration in the list of coder configurations:

coders = cell(CODER_REP_COUNT,length(param_list_coder));
coders_dicts = cell(CODER_REP_COUNT,length(param_list_coder));
sparse_rate = zeros(CODER_REP_COUNT,length(param_list_coder));
final_classifier = cell(CODER_REP_COUNT,length(param_list_coder));
final_labels = cell(CODER_REP_COUNT,length(param_list_coder));
saved_coded_subsample = cell(CODER_REP_COUNT,length(param_list_coder));

coder_scores = zeros(CODER_REP_COUNT,length(param_list_coder));
coder_scores_avg = zeros(1,length(param_list_coder));
coder_scores_std = zeros(1,length(param_list_coder));

coder_build_times = zeros(CODER_REP_COUNT,length(param_list_coder));
coder_build_times_avg = zeros(1,length(param_list_coder));
coder_build_times_std = zeros(1,length(param_list_coder));
coder_code_times = zeros(CODER_REP_COUNT,length(param_list_coder));
coder_code_times_avg = zeros(1,length(param_list_coder));
coder_code_times_std = zeros(1,length(param_list_coder));
coder_classifyfinal_times = zeros(CODER_REP_COUNT,length(param_list_coder));
coder_classifyfinal_times_avg = zeros(1,length(param_list_coder));
coder_classifyfinal_times_std = zeros(1,length(param_list_coder));

logg.beg_node('Testing each coder configuration');

total_time_obj = tic();

for coder_idx = 1:length(param_list_coder)
    logg.beg_node('Configuration %d/%d',coder_idx,length(param_list_coder));

    for coder_rep_idx = 1:CODER_REP_COUNT
        logg.beg_node('Repetition %d/%d',coder_rep_idx,CODER_REP_COUNT);
        
        logg.beg_node('Parameters');
        p_coder_config = rmfield(param_list_coder(coder_idx),{'dictionary_type' 'dictionary_params'});
        p_coder_config.coding_method = param_list_coder(coder_idx).dictionary_params{2};
        p_coder_config.desired_sparsity = param_list_coder(coder_idx).dictionary_params{3}(1);
        p_coder_config.minimum_non_zero = param_list_coder(coder_idx).dictionary_params{3}(2);
        p_coder_config.num_threads = param_list_coder(coder_idx).dictionary_params{3}(3);
        logg.message(utils.params.to_string(p_coder_config));
        logg.end_node();
        
        coder_build_times_obj = tic();
        
        coders{coder_rep_idx,coder_idx} = ...
            transforms.image.recoder(d_coder_useful,...
                                     param_list_coder(coder_idx).patches_count,param_list_coder(coder_idx).window_size,param_list_coder(coder_idx).window_size,0.01,...
                                     param_list_coder(coder_idx).do_patch_zca,param_list_coder(coder_idx).do_polarity_split,...
                                     param_list_coder(coder_idx).dictionary_type,param_list_coder(coder_idx).dictionary_params,...
                                     param_list_coder(coder_idx).window_step,...
                                     param_list_coder(coder_idx).nonlinear_type,param_list_coder(coder_idx).nonlinear_params,...
                                     param_list_coder(coder_idx).reduce_type,param_list_coder(coder_idx).reduce_spread,...
                                     logg.new_node('Training coder'));

        coder_build_times(coder_rep_idx,coder_idx) = toc(coder_build_times_obj);

        coder_code_times_obj = tic();

        d_coder_useful_coded = coders{coder_rep_idx,coder_idx}.code(d_coder_useful,logg.new_node('Coding model selection data'));
        d_ts_coded = coders{coder_rep_idx,coder_idx}.code(d_ts,logg.new_node('Coding final testing data'));

        coders_dicts{coder_rep_idx,coder_idx} = coders{coder_rep_idx,coder_idx}.t_dictionary.dict;
        sparse_rate(coder_rep_idx,coder_idx) = sum(sum(d_coder_useful_coded ~= 0)) / numel(d_coder_useful_coded);

        coder_code_times(coder_rep_idx,coder_idx) = toc(coder_code_times_obj);

        logg.message('Dataset feature count: %d',dataset.geometry(d_coder_useful_coded));
        logg.message('Dataset sparseness: %.2f',sparse_rate(coder_rep_idx,coder_idx));
        
        coder_classifyfinal_times_obj = tic();
        
        final_classifier{coder_rep_idx,coder_idx} = classifiers.svm_linear(d_coder_useful_coded,d_coder_useful_ci,'Primal','L2','L2',CLASSIFIER_REG,'1v1',[TRAIN_WORKER_COUNT CLASSIFY_WORKER_COUNT],logg.new_classifier('Building classifier on full model selection data'));
        [final_labels{coder_rep_idx,coder_idx},~,coder_scores(coder_rep_idx,coder_idx),~,~] = final_classifier{coder_rep_idx,coder_idx}.classify(d_ts_coded,d_ts_ci,logg.new_classifier('Classifying final testing data'));
        
        coder_classifyfinal_times(coder_rep_idx,coder_idx) = toc(coder_classifyfinal_times_obj);
        
        logg.message('Score: %.2f',coder_scores(coder_rep_idx,coder_idx));
        logg.message('Coder build time: %.2fs',coder_build_times(coder_rep_idx,coder_idx));
        logg.message('Coder code time: %.2fs',coder_code_times(coder_rep_idx,coder_idx));
        logg.message('Classification time: %.2fs',coder_classifyfinal_times(coder_rep_idx,coder_idx));
        
        logg.message('Saving intermediate results.');
    
        save(RESULTS_PATH,'-v7.3','coder_idx',...
                                  'MODEL_SELECTION_RATIO','CODER_REP_COUNT',...
                                  'coder_useful_idx',...
                                  'param_desc_coder','param_list_coder',...
                                  'coders','coders_dicts','sparse_rate','final_classifier','final_labels',...
                                  'coder_scores','coder_scores_avg','coder_scores_std');
        
        clear d_coder_useful_coded;
        clear d_ts_coded;
        
        logg.end_node();
    end
    
    coder_scores_avg(coder_idx) = mean(coder_scores(:,coder_idx));
    coder_scores_std(coder_idx) = std(coder_scores(:,coder_idx));
    
    coder_build_times_avg(coder_idx) = mean(coder_build_times(:,coder_idx));
    coder_build_times_std(coder_idx) = std(coder_build_times(:,coder_idx));
    coder_code_times_avg(coder_idx) = mean(coder_code_times(:,coder_idx));
    coder_code_times_std(coder_idx) = std(coder_code_times(:,coder_idx));
    coder_classifyfinal_times_avg(coder_idx) = mean(coder_classifyfinal_times(:,coder_idx));
    coder_classifyfinal_times_std(coder_idx) = std(coder_classifyfinal_times(:,coder_idx));
    
    logg.message('Average score: %.2f +/- %.2f',coder_scores_avg(coder_idx),coder_scores_std(coder_idx));
    logg.message('Average coder build time: %.2fs +/- %.2fs',coder_build_times_avg(coder_idx),coder_build_times_std(coder_idx));
    logg.message('Average coder code time: %.2fs +/- %.2fs',coder_code_times_avg(coder_idx),coder_code_times_std(coder_idx));
    logg.message('Average classification time: %.2fs +/- %.2fs',coder_classifyfinal_times_avg(coder_idx),coder_classifyfinal_times_std(coder_idx));
    
    logg.message('Saving intermediate results.');
    
    save(RESULTS_PATH,'-v7.3','coder_idx',...
                              'MODEL_SELECTION_RATIO','CODER_REP_COUNT',...
                              'coder_useful_idx',...
                              'param_desc_coder','param_list_coder',...
                              'coders','coders_dicts','sparse_rate','final_classifier','final_labels',...
                              'coder_scores','coder_scores_avg','coder_scores_std');

    logg.end_node();
end

total_time = toc(total_time_obj);

logg.end_node();

logg.message('Total search time: %.2fs',total_time);

%% Save results.

logg.message('Saving final results.');

save(RESULTS_PATH,'-v7.3','coder_idx',...
                          'MODEL_SELECTION_RATIO','CODER_REP_COUNT',...
                          'coder_useful_idx',...
                          'param_desc_coder','param_list_coder',...
                          'coders','coders_dicts','sparse_rate','final_classifier','final_labels',...
                          'coder_scores','coder_scores_avg','coder_scores_std');

%% 9.Close the logging subsystem.

logg.end_node();

logg.close();
hnd.close();
