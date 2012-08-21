TESTING_OBSERVATION_COUNT = 20;
PATCH_ROW_COUNT = 9;
PATCH_COL_COUNT = 9;
WORD_COUNT = 128;
NUM_WORKERS = 2;
TEST_COUNT = 2;

param_desc.coding_method = {'Corr' 'MP'};
param_desc.coeff_count = [16 64];
param_desc.nonlinear_type = {'Linear' 'Logistic' 'GlobalOrder'};
param_desc.nonlinear_params = utils.params.condition('nonlinear_type','Linear',[],'Logistic',[],'GlobalOrder',0.01);
param_desc.polarity_split_type = {'None' 'KeepSign'};

param_list = utils.params.gen_all(param_desc);

sample = dataset.load('../../data/mnist.train.mat');
sample = dataset.subsample(sample,1:TESTING_OBSERVATION_COUNT);

times = zeros(TEST_COUNT,length(param_list));
times_avg = zeros(1,length(param_list));
times_std = zeros(1,length(param_list));

for ii = 1:length(param_list)
    fprintf('Configuration %s-%d %s %s\n',param_list(ii).coding_method,param_list(ii).coeff_count,param_list(ii).nonlinear_type,param_list(ii).polarity_split_type);

    codor = transforms.image.recoder(sample,10,PATCH_ROW_COUNT,PATCH_COL_COUNT,0.01,false,...
                                     'Random:Filters',{WORD_COUNT param_list(ii).coding_method [] param_list(ii).coeff_count 1},...
                                     param_list(ii).nonlinear_type,param_list(ii).nonlinear_params,...
                                     param_list(ii).polarity_split_type,'SumSqr',4,NUM_WORKERS);
                                 
    for jj = 1:TEST_COUNT
        timer = tic();

        sample_coded = codor.code(sample);

        times(jj,ii) = toc(timer);
        
        fprintf('  Run #%d/#%d: %.2fs\n',jj,TEST_COUNT,times(jj,ii));
    end
    
    times_avg(ii) = mean(times(:,ii));
    times_std(ii) = std(times(:,ii));
    
    fprintf('  Average time: %.2fs +/- %.2fs\n',times_avg(ii),times_std(ii));
end