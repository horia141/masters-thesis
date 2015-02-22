%% Add "core" to working paths.

addpath ../core

%% Setup experiment-wide constants.

MODEL_SELECTION_RATIO = {[[['full' or unitreal]]] [[['full' or unitreal]]]};
TRAINING_VALIDATION_RATIO = [[[unitreal]]];
CODER_REP_COUNT = [[[natural >= 1]]];
CLASSIFIER_REP_COUNT = [[[natural >= 1];
RESULTS_PATH = '../../explogs/[[[method]]]/[[[dataset]]]/[[[experiment]]]/results_1.mat';
SAVED_SUBSAMPLE_COUNT = [[[natural >= 1]]];

CODER_WORKER_COUNT = 2;
TRAINING_WORKER_COUNT = 2;
CLASSIFY_WORKER_COUNT = 2;

%% Build the list of coder configurations to test.

param_desc_coder.patches_count = 10;
param_desc_coder.patch_dim = [9 11];
param_desc_coder.do_patch_zca = false;
param_desc_coder.dictionary_type = 'Random:Filters';
param_desc_coder.dictionary_params = {{16 'Corr' [] 8 1}};
param_desc_coder.nonlinear_type = 'GlobalOrder';
param_desc_coder.nonlinear_param = 0.01;
param_desc_coder.polarity_split_type = 'None';
param_desc_coder.reduce_type = 'SumSqr';
param_desc_coder.reduce_spread = 4;

param_list_coder = utils.params.gen_all(param_desc_coder);
                        
%% Build the list of classifier configurations to test.

param_desc_classifier.reg = logspace(-2,-1,10);

param_list_classifier = utils.params.gen_all(param_desc_classifier);

%% Start the experiment.

fprintf('Experiment "[[[method]]] - [[[dataset]]] - [[[experiment]]]"\n');

%% Make sure we can write to the results file.

fprintf('  Checking results file.\n');

if exist(RESULTS_PATH,'file')
    error('The results file "%s" already exists!',RESULTS_PATH);
else
    [results_dir,~,~] = fileparts(RESULTS_PATH);
    
    if ~exist(results_dir,'dir')
        fprintf('  Building results file directory.\n');
        system(sprintf('mkdir -p "%s"',results_dir));
    end
end

%% Load the experiment data.

fprintf('  Loading training sample ... ');

[s_tr,ci_tr] = dataset.load('../../data/[[[dataset]]].train.mat');

fprintf('%d observations.\n',dataset.count(s_tr));

fprintf('  Loading test sample ... ');

[s_ts,ci_ts] = dataset.load('../../data/[[[dataset]]].test.mat');

fprintf('%d observations.\n',dataset.count(s_ts));

%% Extract a subsample of the training sample for model selection.

fprintf('  Extracting a subsample of the training sample for model selection ... ');

if check.same(MODEL_SELECTION_RATIO{1},'full')
    N_tr = dataset.count(s_tr);
    coder_useful_idx = 1:N_tr;
    s_coder_useful = s_tr;
    ci_coder_useful = ci_tr;
else
    N_tr = dataset.count(s_tr);
    coder_useful_idx = randi(N_tr,1,min(N_tr,ceil(N_tr * MODEL_SELECTION_RATIO{1})));
    s_coder_useful = dataset.subsample(s_tr,coder_useful_idx);
    ci_coder_useful = ci_tr.subsample(coder_useful_idx);
end

fprintf('%d observations.\n',dataset.count(s_coder_useful));

%% Extract a subsample of the model selection sample for crossvalidation.

fprintf('  Extracting a subsample of the model selection sample for crossvalidation ... ');

if check.same(MODEL_SELECTION_RATIO{2},'full')
    N_coder_useful = dataset.count(s_coder_useful);
    classifier_useful_idx = 1:N_coder_useful;
    ci_classifier_useful = ci_coder_useful;
else
    N_coder_useful = dataset.count(s_coder_useful);
    classifier_useful_idx = randi(N_coder_useful,1,min(N_coder_useful,ceil(N_coder_useful * MODEL_SELECTION_RATIO{2})));
    ci_classifier_useful = ci_coder_useful.subsample(classifier_useful_idx);
end

fprintf('%d observations.\n',length(ci_classifier_useful.labels_idx));

fprintf('  Performing %d splits into training and validation subsamples ... ',CLASSIFIER_REP_COUNT);

training_idx = false(length(classifier_useful_idx),CLASSIFIER_REP_COUNT);
validation_idx = false(length(classifier_useful_idx),CLASSIFIER_REP_COUNT);

for classifier_rep_idx = 1:CLASSIFIER_REP_COUNT
    [training_idx(:,classifier_rep_idx),validation_idx(:,classifier_rep_idx)] = ci_classifier_useful.partition(TRAINING_VALIDATION_RATIO);
end

fprintf('%d training and %d validation observations per split.\n',sum(training_idx(:,1)),sum(validation_idx(:,1)));

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

fprintf('  Testing each coder configuration:\n');

total_time_obj = tic();

for coder_idx = 1:length(param_list_coder)
    fprintf('    Configuration %d/%d:\n',coder_idx,length(param_list_coder));

    for coder_rep_idx = 1:CODER_REP_COUNT
        fprintf('      Repetition %d/%d:\n',coder_rep_idx,CODER_REP_COUNT);
        
        fprintf('        Parameters:\n');
        fprintf('          Patch dim: %d\n',param_list_coder(coder_idx).patch_dim);
        fprintf('          Do patch ZCA: %d\n',param_list_coder(coder_idx).do_patch_zca);
        fprintf('          Dictionary:\n');
        fprintf('            Type: %s\n',param_list_coder(coder_idx).dictionary_type);
        if check.same(param_list_coder(coder_idx).dictionary_type,'Dict')
            fprintf('            Word count: %d\n',size(param_list_coder(coder_idx).dictionary_params{1},1));
        else
            fprintf('            Word count: %d\n',param_list_coder(coder_idx).dictionary_params{1});
        end
        fprintf('            Coding method: %s\n',param_list_coder(coder_idx).dictionary_params{2});
        fprintf('            Coeff count: %d\n',param_list_coder(coder_idx).dictionary_params{4});
        fprintf('          Nonlinear type: %s\n',param_list_coder(coder_idx).nonlinear_type);
        fprintf('          Polarity split type: %s\n',param_list_coder(coder_idx).polarity_split_type);
        fprintf('          Reduce type: %s\n',param_list_coder(coder_idx).reduce_type);
        fprintf('          Reduce spread: %d\n',param_list_coder(coder_idx).reduce_spread);
        
        fprintf('        Building coder.\n');

        coder_build_times_obj = tic();

        coders{coder_rep_idx,coder_idx} = ...
            transforms.image.recoder(s_coder_useful,...
                                     param_list_coder(coder_idx).patches_count,param_list_coder(coder_idx).patch_dim,param_list_coder(coder_idx).patch_dim,0.01,param_list_coder(coder_idx).do_patch_zca,...
                                     param_list_coder(coder_idx).dictionary_type,param_list_coder(coder_idx).dictionary_params,...
                                     param_list_coder(coder_idx).nonlinear_type,param_list_coder(coder_idx).nonlinear_param,...
                                     param_list_coder(coder_idx).polarity_split_type,...
                                     param_list_coder(coder_idx).reduce_type,param_list_coder(coder_idx).reduce_spread,...
                                     CODER_WORKER_COUNT);

        coder_build_times(coder_rep_idx,coder_idx) = toc(coder_build_times_obj);

        fprintf('        Coding model selection and test samples.\n');

        coder_code_times_obj = tic();

        s_coder_useful_coded = coders{coder_rep_idx,coder_idx}.code(s_coder_useful);
        s_ts_coded = coders{coder_rep_idx,coder_idx}.code(s_ts);

        s_classifier_useful_coded = dataset.subsample(s_coder_useful_coded,classifier_useful_idx);

        coders_dicts{coder_rep_idx,coder_idx} = coders{coder_rep_idx,coder_idx}.t_dictionary.dict;
        sparse_rate(coder_rep_idx,coder_idx) = nnz(s_coder_useful_coded) / numel(s_coder_useful_coded);
        saved_coded_subsample{coder_rep_idx,coder_idx} = dataset.subsample(s_coder_useful_coded,1:SAVED_SUBSAMPLE_COUNT);
        saved_coded_subsample_ci{coder_rep_idx,coder_idx} = ci_coder_useful.subsample(1:SAVED_SUBSAMPLE_COUNT);
        
        coder_code_times(coder_rep_idx,coder_idx) = toc(coder_code_times_obj);

        fprintf('        Coded sample feature count: %d\n',dataset.geometry(s_classifier_useful_coded));
        fprintf('        Coded sample sparseness: %.2f\n',sparse_rate(coder_rep_idx,coder_idx));
        
        fprintf('        Testing each classifier configuration:\n');
        
        coder_classifysearch_times_obj = tic();
        
        for classifier_idx = 1:length(param_list_classifier)
            fprintf('          Configuration %d/%d:\n',classifier_idx,length(param_list_classifier));
            
            fprintf('            Parameters:\n');
            fprintf('              Reg: %.4f\n',param_list_classifier(classifier_idx).reg);
            
            for classifier_rep_idx = 1:CLASSIFIER_REP_COUNT
                fprintf('            Repetition %d/%d:\n',classifier_rep_idx,CLASSIFIER_REP_COUNT);
                
                classifier_times_obj = tic();
                
                s_training_coded = dataset.subsample(s_classifier_useful_coded,training_idx(:,classifier_rep_idx)');
                ci_training = ci_classifier_useful.subsample(training_idx(:,classifier_rep_idx));
                d_validation_coded = dataset.subsample(s_classifier_useful_coded,validation_idx(:,classifier_rep_idx)');
                ci_validation = ci_classifier_useful.subsample(validation_idx(:,classifier_rep_idx));
                
                fprintf('              Training SVM on current training sample and evaluating on current validation sample.\n');

                cl = classifiers.linear.svm(s_training_coded,ci_training,'Primal','L2','L2',param_list_classifier(classifier_idx).reg,'1va',[TRAINING_WORKER_COUNT CLASSIFY_WORKER_COUNT]);
                [~,~,classifier_scores(classifier_rep_idx,classifier_idx,coder_rep_idx,coder_idx),~,~] = cl.classify(d_validation_coded,ci_validation);
                
                classifier_times(classifier_rep_idx,classifier_idx,coder_rep_idx,coder_idx) = toc(classifier_times_obj);

                fprintf('              Score: %.2f%%\n',classifier_scores(classifier_rep_idx,classifier_idx,coder_rep_idx,coder_idx));
                fprintf('              Time: %.2fs\n',classifier_times(classifier_rep_idx,classifier_idx,coder_rep_idx,coder_idx));
            end
            
            classifier_scores_avg(classifier_idx,coder_rep_idx,coder_idx) = mean(classifier_scores(:,classifier_idx,coder_rep_idx,coder_idx));
            classifier_scores_std(classifier_idx,coder_rep_idx,coder_idx) = std(classifier_scores(:,classifier_idx,coder_rep_idx,coder_idx));
            
            classifier_times_per_classifier_avg(classifier_idx,coder_rep_idx,coder_idx) = mean(classifier_times(:,classifier_idx,coder_rep_idx,coder_idx));
            classifier_times_per_classifier_std(classifier_idx,coder_rep_idx,coder_idx) = std(classifier_times(:,classifier_idx,coder_rep_idx,coder_idx));

            fprintf('            Average score: %.2f%% +/- %.2f\n',classifier_scores_avg(classifier_idx,coder_rep_idx,coder_idx),classifier_scores_std(classifier_idx,coder_rep_idx,coder_idx));
            fprintf('            Average time: %.2fs +/- %.2fs\n',classifier_times_per_classifier_avg(classifier_idx,coder_rep_idx,coder_idx),classifier_times_per_classifier_std(classifier_idx,coder_rep_idx,coder_idx));
            
            if classifier_scores_avg(classifier_idx,coder_rep_idx,coder_idx) > best_classifier_score_avg(coder_rep_idx,coder_idx)
                fprintf('            New best configuration!\n');
                
                if best_classifier_idx(coder_rep_idx,coder_idx) ~= -1
                    fprintf('            Previous best configuration:\n');
                    fprintf('              %s\n',utils.params.to_string(param_list_classifier(best_classifier_idx(coder_rep_idx,coder_idx))));
                    fprintf('            Previous best score: %.2f%%\n',best_classifier_score_avg(coder_rep_idx,coder_idx));
                end

                best_classifier_score_avg(coder_rep_idx,coder_idx) = classifier_scores_avg(classifier_idx,coder_rep_idx,coder_idx);
                best_classifier_idx(coder_rep_idx,coder_idx) = classifier_idx;
            else
                fprintf('            Current best configuration:\n');
                fprintf('              %s\n',utils.params.to_string(param_list_classifier(best_classifier_idx(coder_rep_idx,coder_idx))));
                fprintf('            Current best score: %.2f%%\n',best_classifier_score_avg(coder_rep_idx,coder_idx));
            end
        end
        
        coder_classifysearch_times(coder_rep_idx,coder_idx) = toc(coder_classifysearch_times_obj);
        
        coder_classifyfinal_times_obj = tic();
        
        fprintf('        Training with best classifier configuration on model selection sample and evaluating on test sample.\n');
        
        final_classifier{coder_rep_idx,coder_idx} = classifiers.linear.svm(s_coder_useful_coded,ci_coder_useful,'Primal','L2','L2',param_list_classifier(best_classifier_idx(coder_rep_idx,coder_idx)).reg,'1va',[TRAINING_WORKER_COUNT CLASSIFY_WORKER_COUNT]);
        [final_labels{coder_rep_idx,coder_idx},~,coder_scores(coder_rep_idx,coder_idx),~,~] = final_classifier{coder_rep_idx,coder_idx}.classify(s_ts_coded,ci_ts);
        
        coder_classifyfinal_times(coder_rep_idx,coder_idx) = toc(coder_classifyfinal_times_obj);
        
        classifier_times_per_coder_rep_avg(coder_rep_idx,coder_idx) = mean(reshape(classifier_times(:,:,coder_rep_idx,coder_idx),1,[]));
        classifier_times_per_coder_rep_std(coder_rep_idx,coder_idx) = std(reshape(classifier_times(:,:,coder_rep_idx,coder_idx),1,[]));

        fprintf('        Final Score: %.2f%%\n',coder_scores(coder_rep_idx,coder_idx));
        fprintf('        Coder build time: %.2fs\n',coder_build_times(coder_rep_idx,coder_idx));
        fprintf('        Coder code time: %.2fs\n',coder_code_times(coder_rep_idx,coder_idx));
        fprintf('        Classifier search time: %.2fs\n',coder_classifysearch_times(coder_rep_idx,coder_idx));
        fprintf('        Final classification time: %.2fs\n',coder_classifyfinal_times(coder_rep_idx,coder_idx));
        fprintf('        Average classifier time: %.2fs +/- %.2fs\n',classifier_times_per_coder_rep_avg(coder_rep_idx,coder_idx),classifier_times_per_coder_rep_std(coder_rep_idx,coder_idx));
        
        fprintf('        Saving intermediate results.\n');
    
        save(RESULTS_PATH,'-v7.3','coder_idx','coder_rep_idx',...
                                  'MODEL_SELECTION_RATIO','TRAINING_VALIDATION_RATIO','CODER_REP_COUNT','CLASSIFIER_REP_COUNT','RESULTS_PATH','SAVED_SUBSAMPLE_COUNT',...
                                  'coder_useful_idx','classifier_useful_idx','training_idx','validation_idx',...
                                  'param_desc_coder','param_list_coder','param_desc_classifier','param_list_classifier',...
                                  'coders','coders_dicts','sparse_rate','saved_coded_subsample','saved_coded_subsample_ci','final_classifier','final_labels',...
                                  'classifier_scores','classifier_scores_avg','classifier_scores_std',...
                                  'best_classifier_score_avg','best_classifier_idx',...
                                  'coder_scores','coder_scores_avg','coder_scores_std');
        
        clear s_coder_useful_coded;
        clear s_ts_coded;
        clear s_classifier_useful_coded;
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
    
    fprintf('      Average final score: %.2f%% +/- %.2f\n',coder_scores_avg(coder_idx),coder_scores_std(coder_idx));
    fprintf('      Average coder build time: %.2fs +/- %.2fs\n',coder_build_times_avg(coder_idx),coder_build_times_std(coder_idx));
    fprintf('      Average coder code time: %.2fs +/- %.2fs\n',coder_code_times_avg(coder_idx),coder_code_times_std(coder_idx));
    fprintf('      Average classifier search time: %.2fs +/- %.2fs\n',coder_classifysearch_times_avg(coder_idx),coder_classifysearch_times_std(coder_idx));
    fprintf('      Average final classification time: %.2fs +/- %.2fs\n',coder_classifyfinal_times_avg(coder_idx),coder_classifyfinal_times_std(coder_idx));
    fprintf('      Average classifier time: %.2fs +/- %.2fs\n',classifier_times_per_coder_avg(coder_idx),classifier_times_per_coder_std(coder_idx));
    
    fprintf('      Saving intermediate results.\n');
    
    save(RESULTS_PATH,'-v7.3','coder_idx',...
                              'MODEL_SELECTION_RATIO','TRAINING_VALIDATION_RATIO','CODER_REP_COUNT','CLASSIFIER_REP_COUNT','RESULTS_PATH','SAVED_SUBSAMPLE_COUNT',...
                              'coder_useful_idx','classifier_useful_idx','training_idx','validation_idx',...
                              'param_desc_coder','param_list_coder','param_desc_classifier','param_list_classifier',...
                              'coders','coders_dicts','sparse_rate','saved_coded_subsample','saved_coded_subsample_ci','final_classifier','final_labels',...
                              'classifier_scores','classifier_scores_avg','classifier_scores_std',...
                              'best_classifier_score_avg','best_classifier_idx',...
                              'coder_scores','coder_scores_avg','coder_scores_std');
end

total_time = toc(total_time_obj);

fprintf('  Total search time: %.2fs\n',total_time);

%% Save results.

fprintf('  Saving final results.\n');

save(RESULTS_PATH,'-v7.3','MODEL_SELECTION_RATIO','TRAINING_VALIDATION_RATIO','CODER_REP_COUNT','CLASSIFIER_REP_COUNT','RESULTS_PATH','SAVED_SUBSAMPLE_COUNT',...
                          'coder_useful_idx','training_idx','validation_idx',...
                          'param_desc_coder','param_list_coder','param_desc_classifier','param_list_classifier',...
                          'coders','coders_dicts','sparse_rate','saved_coded_subsample','saved_coded_subsample_ci','final_classifier','final_labels',...
                          'classifier_scores','classifier_scores_avg','classifier_scores_std',...
                          'best_classifier_score_avg','best_classifier_idx',...
                          'coder_scores','coder_scores_avg','coder_scores_std');
