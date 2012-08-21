%% Setup experiment-wide constants.

MODEL_SELECTION_RATIO = {'full' 0.2};
TRAIN_VALIDATION_RATIO = 0.5;
CODER_REP_COUNT = 1;
CLASSIFIER_REP_COUNT = 5;
RESULTS_PATH = '../explogs/smallnorb/sparse_coding_neural_gas/test_1/results_3_CCmp11_w1024_p13_k11_s3_zca_dict.mat';
SAVED_SUBSAMPLE_COUNT = 20;

TRAIN_WORKER_COUNT = 45;
CLASSIFY_WORKER_COUNT = 48;

%% Build the list of coder configurations to test.

load ../explogs/smallnorb/sparse_coding_neural_gas/saved_dict_mp17_13x13_1024.mat
size(saved_dict)

% Coding method.
param_desc_coder.patches_count = 10;
param_desc_coder.do_patch_zca = false;
param_desc_coder.dictionary_type = 'Dict';
param_desc_coder.dictionary_params = {{saved_dict 'MP' 17}};
% Coder transforms.
param_desc_coder.do_polarity_split = false;
param_desc_coder.nonlinear_type = 'Logistic';
param_desc_coder.nonlinear_params = {};
param_desc_coder.reduce_type = 'Sqr';
% Coder geometry.
param_desc_coder.window_size = 13;
param_desc_coder.window_step = 3;
param_desc_coder.reduce_spread = 11;

param_list_coder = utils.params.gen_all(param_desc_coder,...
                                        @(p)mod(97 - 1,p.window_step) == 0,...
                                        @(p)mod((97 - 1) / p.window_step + 1,p.reduce_spread) == 0);
                        
%% Build the list of classifier configurations to test.

param_desc_classifier.reg = logspace(-2,0,20);

param_list_classifier = utils.params.gen_all(param_desc_classifier);

%% Initialize the logging subsystem.

hnd = logging.handlers.stdout(logging.level.Experiment);
logg = logging.logger({hnd});

logg.beg_node('Experiment "SmallNORB - Sparse Coding Neural Gas - Test 1"');

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

[d_tr_t,d_tr_ci,d_ts_t,d_ts_ci] = utils.load_dataset.smallnorb(logg.new_node('Loading SmallNORB dataset'));
d_tr = zeros(97,97,1,dataset.count(d_tr_t));
d_tr(1:96,1:96,1,:) = d_tr_t(:,:,1,:);
d_ts = zeros(97,97,1,dataset.count(d_ts_t));
d_ts(1:96,1:96,1,:) = d_ts_t(:,:,1,:);

%% Filter the experiment model selection data, retaining only a subsample from it.

logg.message('Extracting the data for model selection.');

if check.same(MODEL_SELECTION_RATIO{1},'full')
    N_tr = dataset.count(d_tr);
    coder_useful_idx = 1:N_tr;
    d_coder_useful = d_tr;
    d_coder_useful_ci = d_tr_ci;
else
    N_tr = dataset.count(d_tr);
    coder_useful_idx = randi(N_tr,1,min(N_tr,ceil(N_tr * MODEL_SELECTION_RATIO{1})));
    d_coder_useful = dataset.subsample(d_tr,coder_useful_idx);
    d_coder_useful_ci = d_tr_ci.subsample(coder_useful_idx);
end

if check.same(MODEL_SELECTION_RATIO{2},'full')
    N_coder_useful = dataset.count(d_coder_useful);
    classifier_useful_idx = 1:N_coder_useful;
    d_classifier_useful_ci = d_coder_useful_ci;
else
    N_coder_useful = dataset.count(d_coder_useful);
    classifier_useful_idx = randi(N_coder_useful,1,min(N_coder_useful,ceil(N_coder_useful * MODEL_SELECTION_RATIO{2})));
    d_classifier_useful_ci = d_coder_useful_ci.subsample(classifier_useful_idx);
end

%% For each repetition of the classifier configuration in [1:CLASSIFIER_REP_COUNT].

logg.message('Building train/validation indices.');

train_idx = false(length(classifier_useful_idx),CLASSIFIER_REP_COUNT);
validation_idx = false(length(classifier_useful_idx),CLASSIFIER_REP_COUNT);

for classifier_rep_idx = 1:CLASSIFIER_REP_COUNT
    [train_idx(:,classifier_rep_idx),validation_idx(:,classifier_rep_idx)] = d_classifier_useful_ci.partition(TRAIN_VALIDATION_RATIO);
end

%% For each coder configuration in the list of coder configurations:

coders = cell(CODER_REP_COUNT,length(param_list_coder));
coders_dicts = cell(CODER_REP_COUNT,length(param_list_coder));
sparse_rate = zeros(CODER_REP_COUNT,length(param_list_coder));
saved_coded_subsample = cell(CODER_REP_COUNT,length(param_list_coder));
saved_coded_subsample_ci = cell(CODER_REP_COUNT,length(param_list_coder));
final_classifier = cell(CODER_REP_COUNT,length(param_list_coder));
final_labels = cell(CODER_REP_COUNT,length(param_list_coder));

classifier_scores = zeros(CLASSIFIER_REP_COUNT,length(param_list_classifier),CODER_REP_COUNT,length(param_list_coder));
classifier_scores_avg = zeros(length(param_list_classifier),CODER_REP_COUNT,length(param_list_coder));
classifier_scores_std = zeros(length(param_list_classifier),CODER_REP_COUNT,length(param_list_coder));
best_classifier_score_avg = -inf * ones(CODER_REP_COUNT,length(param_list_coder));
best_classifier_idx = -1 * ones(CODER_REP_COUNT,length(param_list_coder));
coder_scores = zeros(CODER_REP_COUNT,length(param_list_coder));
coder_scores_avg = zeros(1,length(param_list_coder));
coder_scores_std = zeros(1,length(param_list_coder));

classifier_times = zeros(CLASSIFIER_REP_COUNT,length(param_list_classifier),CODER_REP_COUNT,length(param_list_coder));
classifier_times_per_classifier_avg = zeros(length(param_list_classifier),CODER_REP_COUNT,length(param_list_coder));
classifier_times_per_classifier_std = zeros(length(param_list_classifier),CODER_REP_COUNT,length(param_list_coder));
classifier_times_per_coder_rep_avg = zeros(CODER_REP_COUNT,length(param_list_coder));
classifier_times_per_coder_rep_std = zeros(CODER_REP_COUNT,length(param_list_coder));
classifier_times_per_coder_avg = zeros(1,length(param_list_coder));
classifier_times_per_coder_std = zeros(1,length(param_list_coder));
coder_build_times = zeros(CODER_REP_COUNT,length(param_list_coder));
coder_build_times_avg = zeros(1,length(param_list_coder));
coder_build_times_std = zeros(1,length(param_list_coder));
coder_code_times = zeros(CODER_REP_COUNT,length(param_list_coder));
coder_code_times_avg = zeros(1,length(param_list_coder));
coder_code_times_std = zeros(1,length(param_list_coder));
coder_classifysearch_times = zeros(CODER_REP_COUNT,length(param_list_coder));
coder_classifysearch_times_avg = zeros(1,length(param_list_coder));
coder_classifysearch_times_std = zeros(1,length(param_list_coder));
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
        p_coder_config.coding_params = param_list_coder(coder_idx).dictionary_params{3};
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

        d_classifier_useful_coded = dataset.subsample(d_coder_useful_coded,classifier_useful_idx);

        coders_dicts{coder_rep_idx,coder_idx} = coders{coder_rep_idx,coder_idx}.t_dictionary.dict;
        sparse_rate(coder_rep_idx,coder_idx) = sum(sum(d_coder_useful_coded ~= 0)) / numel(d_coder_useful_coded);
        saved_coded_subsample{coder_rep_idx,coder_idx} = dataset.subsample(d_coder_useful_coded,1:SAVED_SUBSAMPLE_COUNT);
        saved_coded_subsample_ci{coder_rep_idx,coder_idx} = d_coder_useful_ci.subsample(1:SAVED_SUBSAMPLE_COUNT);
        
        coder_code_times(coder_rep_idx,coder_idx) = toc(coder_code_times_obj);

        logg.message('Crossvalidation dataset size: %d',dataset.count(d_classifier_useful_coded));
        logg.message('Test dataset(s) size: %d',ceil(TRAIN_VALIDATION_RATIO * dataset.count(d_classifier_useful_coded)));
        logg.message('Validation dataset(s) size: %d',ceil((1 - TRAIN_VALIDATION_RATIO) * dataset.count(d_classifier_useful_coded)));
        logg.message('Dataset feature count: %d',dataset.geometry(d_classifier_useful_coded));
        logg.message('Dataset sparseness: %.2f',sparse_rate(coder_rep_idx,coder_idx));
        
        logg.beg_node('Testing each classifier configuration');
        
        coder_classifysearch_times_obj = tic();
        
        for classifier_idx = 1:length(param_list_classifier)
            logg.beg_node('Configuration %d/%d',classifier_idx,length(param_list_classifier));
            
            logg.beg_node('Parameters');
            logg.message(utils.params.to_string(param_list_classifier(classifier_idx)));
            logg.end_node();
            
            for classifier_rep_idx = 1:CLASSIFIER_REP_COUNT
                logg.beg_node('Repetition %d/%d',classifier_rep_idx,CLASSIFIER_REP_COUNT);
                
                classifier_times_obj = tic();
                
                d_train_coded = dataset.subsample(d_classifier_useful_coded,train_idx(:,classifier_rep_idx)');
                d_train_ci = d_classifier_useful_ci.subsample(train_idx(:,classifier_rep_idx));
                d_validation_coded = dataset.subsample(d_classifier_useful_coded,validation_idx(:,classifier_rep_idx)');
                d_validation_ci = d_classifier_useful_ci.subsample(validation_idx(:,classifier_rep_idx));
                
                cl = classifiers.svm_linear(d_train_coded,d_train_ci,'Primal','L2','L2',param_list_classifier(classifier_idx).reg,'1v1',[TRAIN_WORKER_COUNT CLASSIFY_WORKER_COUNT],logg.new_classifier('Building classifier on training subsample'));
                [~,~,classifier_scores(classifier_rep_idx,classifier_idx,coder_rep_idx,coder_idx),~,~] = cl.classify(d_validation_coded,d_validation_ci,logg.new_classifier('Classifying validation subsample'));
                
                classifier_times(classifier_rep_idx,classifier_idx,coder_rep_idx,coder_idx) = toc(classifier_times_obj);

                logg.message('Score: %.2f',classifier_scores(classifier_rep_idx,classifier_idx,coder_rep_idx,coder_idx));
                logg.message('Time: %.2fs',classifier_times(classifier_rep_idx,classifier_idx,coder_rep_idx,coder_idx));

                logg.end_node();
            end
            
            classifier_scores_avg(classifier_idx,coder_rep_idx,coder_idx) = mean(classifier_scores(:,classifier_idx,coder_rep_idx,coder_idx));
            classifier_scores_std(classifier_idx,coder_rep_idx,coder_idx) = std(classifier_scores(:,classifier_idx,coder_rep_idx,coder_idx));
            
            classifier_times_per_classifier_avg(classifier_idx,coder_rep_idx,coder_idx) = mean(classifier_times(:,classifier_idx,coder_rep_idx,coder_idx));
            classifier_times_per_classifier_std(classifier_idx,coder_rep_idx,coder_idx) = std(classifier_times(:,classifier_idx,coder_rep_idx,coder_idx));

            logg.message('Average score: %.2f +/- %.2f',classifier_scores_avg(classifier_idx,coder_rep_idx,coder_idx),classifier_scores_std(classifier_idx,coder_rep_idx,coder_idx));
            logg.message('Average time: %.2fs +/- %.2fs',classifier_times_per_classifier_avg(classifier_idx,coder_rep_idx,coder_idx),classifier_times_per_classifier_std(classifier_idx,coder_rep_idx,coder_idx));
            
            if classifier_scores_avg(classifier_idx,coder_rep_idx,coder_idx) > best_classifier_score_avg(coder_rep_idx,coder_idx)
                logg.message('New best configuration!');
                
                if best_classifier_idx(coder_rep_idx,coder_idx) ~= -1
                    logg.beg_node('Previous best configuration');
                    logg.message(utils.params.to_string(param_list_classifier(best_classifier_idx(coder_rep_idx,coder_idx))));
                    logg.end_node();
                    logg.message('Previous best score: %.2f',best_classifier_score_avg(coder_rep_idx,coder_idx));
                end

                best_classifier_score_avg(coder_rep_idx,coder_idx) = classifier_scores_avg(classifier_idx,coder_rep_idx,coder_idx);
                best_classifier_idx(coder_rep_idx,coder_idx) = classifier_idx;
            else
                logg.beg_node('Current best configuration');
                logg.message(utils.params.to_string(param_list_classifier(best_classifier_idx(coder_rep_idx,coder_idx))));
                logg.end_node();
                logg.message('Current best score: %.2f',best_classifier_score_avg(coder_rep_idx,coder_idx));
            end
            
            logg.end_node();
        end
        
        coder_classifysearch_times(coder_rep_idx,coder_idx) = toc(coder_classifysearch_times_obj);
        
        logg.end_node();
        
        coder_classifyfinal_times_obj = tic();
        
        final_classifier{coder_rep_idx,coder_idx} = classifiers.svm_linear(d_coder_useful_coded,d_coder_useful_ci,'Primal','L2','L2',param_list_classifier(best_classifier_idx(coder_rep_idx,coder_idx)).reg,'1v1',[TRAIN_WORKER_COUNT CLASSIFY_WORKER_COUNT],logg.new_classifier('Building classifier on full model selection data'));
        [final_labels{coder_rep_idx,coder_idx},~,coder_scores(coder_rep_idx,coder_idx),~,~] = final_classifier{coder_rep_idx,coder_idx}.classify(d_ts_coded,d_ts_ci,logg.new_classifier('Classifying final testing data'));
        
        coder_classifyfinal_times(coder_rep_idx,coder_idx) = toc(coder_classifyfinal_times_obj);
        
        classifier_times_per_coder_rep_avg(coder_rep_idx,coder_idx) = mean(reshape(classifier_times(:,:,coder_rep_idx,coder_idx),1,[]));
        classifier_times_per_coder_rep_std(coder_rep_idx,coder_idx) = std(reshape(classifier_times(:,:,coder_rep_idx,coder_idx),1,[]));

        logg.message('Final Score: %.2f',coder_scores(coder_rep_idx,coder_idx));
        logg.message('Coder build time: %.2fs',coder_build_times(coder_rep_idx,coder_idx));
        logg.message('Coder code time: %.2fs',coder_code_times(coder_rep_idx,coder_idx));
        logg.message('Classifier search time: %.2fs',coder_classifysearch_times(coder_rep_idx,coder_idx));
        logg.message('Final classification time: %.2fs',coder_classifyfinal_times(coder_rep_idx,coder_idx));
        logg.message('Average classifier time: %.2fs +/- %.2fs',classifier_times_per_coder_rep_avg(coder_rep_idx,coder_idx),classifier_times_per_coder_rep_std(coder_rep_idx,coder_idx));
        
        logg.message('Saving intermediate results.');
    
        save(RESULTS_PATH,'-v7.3','coder_idx',...
                                  'MODEL_SELECTION_RATIO','TRAIN_VALIDATION_RATIO','CODER_REP_COUNT','CLASSIFIER_REP_COUNT','RESULTS_PATH','SAVED_SUBSAMPLE_COUNT',...
                                  'coder_useful_idx','classifier_useful_idx','train_idx','validation_idx',...
                                  'param_desc_coder','param_list_coder','param_desc_classifier','param_list_classifier',...
                                  'coders','coders_dicts','sparse_rate','saved_coded_subsample','saved_coded_subsample_ci','final_classifier','final_labels',...
                                  'classifier_scores','classifier_scores_avg','classifier_scores_std',...
                                  'best_classifier_score_avg','best_classifier_idx',...
                                  'coder_scores','coder_scores_avg','coder_scores_std');
        
        clear d_coder_useful_coded;
        clear d_ts_coded;
        clear d_classifier_useful_coded;
        
        logg.end_node();
    end
    
    coder_scores_avg(coder_idx) = mean(coder_scores(:,coder_idx));
    coder_scores_std(coder_idx) = std(coder_scores(:,coder_idx));
    
    coder_build_times_avg(coder_idx) = mean(coder_build_times(:,coder_idx));
    coder_build_times_std(coder_idx) = std(coder_build_times(:,coder_idx));
    coder_code_times_avg(coder_idx) = mean(coder_code_times(:,coder_idx));
    coder_code_times_std(coder_idx) = std(coder_code_times(:,coder_idx));
    coder_classifysearch_times_avg(coder_idx) = mean(coder_classifysearch_times(:,coder_idx));
    coder_classifysearch_times_std(coder_idx) = std(coder_classifysearch_times(:,coder_idx));
    coder_classifyfinal_times_avg(coder_idx) = mean(coder_classifyfinal_times(:,coder_idx));
    coder_classifyfinal_times_std(coder_idx) = std(coder_classifyfinal_times(:,coder_idx));
    classifier_times_per_coder_avg(coder_idx) = mean(reshape(classifier_times(:,:,:,coder_idx),1,[]));
    classifier_times_per_coder_std(coder_idx) = std(reshape(classifier_times(:,:,:,coder_idx),1,[]));
    
    logg.message('Average final score: %.2f +/- %.2f',coder_scores_avg(coder_idx),coder_scores_std(coder_idx));
    logg.message('Average coder build time: %.2fs +/- %.2fs',coder_build_times_avg(coder_idx),coder_build_times_std(coder_idx));
    logg.message('Average coder code time: %.2fs +/- %.2fs',coder_code_times_avg(coder_idx),coder_code_times_std(coder_idx));
    logg.message('Average classifier search time: %.2fs +/- %.2fs',coder_classifysearch_times_avg(coder_idx),coder_classifysearch_times_std(coder_idx));
    logg.message('Average final classification time: %.2fs +/- %.2fs',coder_classifyfinal_times_avg(coder_idx),coder_classifyfinal_times_std(coder_idx));
    logg.message('Average classifier time: %.2fs +/- %.2fs',classifier_times_per_coder_avg(coder_idx),classifier_times_per_coder_std(coder_idx));
    
    logg.message('Saving intermediate results.');
    
    save(RESULTS_PATH,'-v7.3','coder_idx',...
                              'MODEL_SELECTION_RATIO','TRAIN_VALIDATION_RATIO','CODER_REP_COUNT','CLASSIFIER_REP_COUNT','RESULTS_PATH','SAVED_SUBSAMPLE_COUNT',...
                              'coder_useful_idx','classifier_useful_idx','train_idx','validation_idx',...
                              'param_desc_coder','param_list_coder','param_desc_classifier','param_list_classifier',...
                              'coders','coders_dicts','sparse_rate','saved_coded_subsample','saved_coded_subsample_ci','final_classifier','final_labels',...
                              'classifier_scores','classifier_scores_avg','classifier_scores_std',...
                              'best_classifier_score_avg','best_classifier_idx',...
                              'coder_scores','coder_scores_avg','coder_scores_std');
                          
    logg.end_node();
end

total_time = toc(total_time_obj);

logg.end_node();

logg.message('Total search time: %.2fs',total_time);

%% Save results.

logg.message('Saving final results.');

save(RESULTS_PATH,'-v7.3','MODEL_SELECTION_RATIO','TRAIN_VALIDATION_RATIO','CODER_REP_COUNT','CLASSIFIER_REP_COUNT','RESULTS_PATH','SAVED_SUBSAMPLE_COUNT',...
                          'coder_useful_idx','train_idx','validation_idx',...
                          'param_desc_coder','param_list_coder','param_desc_classifier','param_list_classifier',...
                          'coders','coders_dicts','sparse_rate','saved_coded_subsample','saved_coded_subsample_ci','final_classifier','final_labels',...
                          'classifier_scores','classifier_scores_avg','classifier_scores_std',...
                          'best_classifier_score_avg','best_classifier_idx',...
                          'coder_scores','coder_scores_avg','coder_scores_std');

%% 9.Close the logging subsystem.

logg.end_node();

logg.close();
hnd.close();
