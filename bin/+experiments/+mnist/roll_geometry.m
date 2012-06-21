%% Setup experiment.

FOLD_COUNT = 10;
RESULTS_PATH = '../explogs/mnist/roll_geometry/results.mat';
TRAIN_WORKERS_COUNT = 45;
CLASSIFY_WORKERS_COUNT = 48;

param_desc.dictionary_type = 'Random:Filters';
param_desc.dictionary_params = {{100 'Corr' {}}};
param_desc.window_size = [3 5 7 9 11 13 15 17];
param_desc.window_step = [1 3 9];
param_desc.nonlinear_type = 'Logistic';
param_desc.nonlinear_params = {};
param_desc.reduce_type = 'Sqr';
param_desc.reduce_spread = [2 4 5 7 10 14 28];

param_list = params.gen_all(param_desc,...
                            @(p)mod(28 - 1,p.window_step) == 0,...
                            @(p)mod((28 - 1) / p.window_step + 1,p.reduce_spread) == 0);

param_desc_class.reg = logspace(-1.5,0.3,10);
param_list_class = params.gen_all(param_desc_class);

hnd = logging.handlers.stdout(logging.level.Experiment);
logg = logging.logger({hnd});

%% Start experiment.

logg.beg_experiment('Experiment "MNIST - Roll Geometry"');

[m_tr,m_tr_ci,m_ts,m_ts_ci] = utilstest.load_mnist(logg.new_node('Loading MNIST dataset'));

alphas = cell(17,9,28);
t_wsrs = cell(17,9,28);
cls_best = cell(17,9,28);
scores = zeros(17,9,28);

logg.beg_node('Evaluating classifier performance for each configuration');

for ii = 1:length(param_list)
    logg.beg_node('Configuration %d/%d',ii,length(param_list));
    
    logg.beg_node('Parameters');
    p_param_list = rmfield(param_list(ii),{'dictionary_type' 'dictionary_params'});
    logg.message(params.to_string(p_param_list));
    logg.end_node();
    
    ca = param_list(ii).window_size;
    cb = param_list(ii).window_step;
    cc = param_list(ii).reduce_spread;
    
    [alphas{ca,cb,cc},~,t_wsrs{ca,cb,cc},cls_best{ca,cb,cc},~,~,~,~,scores(ca,cb,cc),~,~] = ...
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
    load ../explogs/mnist/roll_geometry/results.mat
end

figure;
subplot(2,3,1);
[a,~,c] = find(squeeze(scores(:,1,2)));
plot(a,c);
axis([1 17 90 100]);
xlabel('Window Size');
title('For Reduce Spread = 2');
subplot(2,3,2);
[a,~,c] = find(squeeze(scores(:,1,4)));
plot(a,c);
axis([1 17 90 100]);
xlabel('Window Size');
title('For Reduce Spread = 4');
subplot(2,3,3);
[a,~,c] = find(squeeze(scores(:,1,7)));
plot(a,c);
axis([1 17 90 100]);
xlabel('Window Size');
title('For Reduce Spread = 7');
subplot(2,3,4);
[a,~,c] = find(squeeze(scores(:,1,14)));
plot(a,c);
axis([1 17 90 100]);
xlabel('Window Size');
title('For Reduce Spread = 14');
subplot(2,3,5);
[a,~,c] = find(squeeze(scores(:,1,28)));
plot(a,c);
axis([1 17 90 100]);
xlabel('Window Size');
title('For Reduce Spread = 28');

figure;
subplot(2,2,1);
[a,~,c] = find(squeeze(scores(:,3,2)));
plot(a,c);
axis([1 17 90 100]);
xlabel('Window Size');
title('For Reduce Spread = 2');
subplot(2,2,2);
[a,~,c] = find(squeeze(scores(:,3,5)));
plot(a,c);
axis([1 17 90 100]);
xlabel('Window Size');
title('For Reduce Spread = 5');
subplot(2,2,3);
[a,~,c] = find(squeeze(scores(:,3,10)));
plot(a,c);
axis([1 17 50 100]);
xlabel('Window Size');
title('For Reduce Spread = 10');

figure;
subplot(2,1,1);
[a,~,c] = find(squeeze(scores(:,9,2)));
plot(a,c);
axis([1 17 50 100]);
xlabel('Window Size');
title('For Reduce Spread = 2');
subplot(2,1,2);
[a,~,c] = find(squeeze(scores(:,9,4)));
plot(a,c);
axis([1 17 50 100]);
xlabel('Window Size');
title('For Reduce Spread = 4');

figure;
subplot(1,3,1);
bar3(squeeze(scores(:,1,:)));
axis([1 28 1 17 0 100]);
xlabel('Reduce Spread');
ylabel('Window Size');
title('For Window Step = 1');
subplot(1,3,2);
bar3(squeeze(scores(:,3,:)));
axis([1 28 1 17 0  100]);
xlabel('Reduce Spread');
ylabel('Window Size');
title('For Window Step = 3');
subplot(1,3,3);
bar3(squeeze(scores(:,9,:)));
axis([1 28 1 17 0  100]);
xlabel('Reduce Spread');
ylabel('Window Size');
title('For Window Step = 9');