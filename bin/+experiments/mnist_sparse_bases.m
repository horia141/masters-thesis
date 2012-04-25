%% Setup experiment.

PATCHES_COUNT = 100000;
PATCH_ROW_COUNT = 12;
PATCH_COL_COUNT = 12;
PATCH_MIN_VARIANCE = 0.02;
DETAILS_LOGFILE_PATH = sprintf('../explogs/mnist_sparse_bases_just_coding_%s.log',datestr(now,'yyyy-mm-dd_HH-MM'));

params_desc.coding_fn = {@transforms.sparse.gdmp.correlation @transforms.sparse.gdmp.matching_pursuit @transforms.sparse.gdmp.ortho_matching_pursuit};
params_desc.sparse_word_count = [64 81 100 121 144 169 192 225 256];
params_desc.sparse_coeffs_count = [1:10 15 20];

params_list = params.gen_all(params_desc);

hnd_overview = logging.handlers.stdout(logging.level.Experiment);
hnd_details = logging.handlers.file(DETAILS_LOGFILE_PATH,logging.level.All);
log = logging.logger({hnd_overview hnd_details});

log.beg_experiment('MNIST: Train sparse dictionaries on 8x8 patches');

%% Load data and do initial pre-processing.

log.beg_node('Loading data and extracting patches');

mnist_raw = datasets.image.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte',log.new_dataset_io('Loading MNIST training data'));
t_patches = transforms.image.patch_extract(mnist_raw,PATCHES_COUNT,PATCH_ROW_COUNT,PATCH_COL_COUNT,PATCH_MIN_VARIANCE,log.new_transform('Building patch extract transform'));
mnist_1 = t_patches.code(mnist_raw,log.new_transform('Extracting patches'));

log.end_node();

log.beg_node('Pre-processing patches');

t_window = transforms.image.window(mnist_1,4.5,log.new_transform('Building window transform'));
mnist_2 = t_window.code(mnist_1,log.new_transform('Applying window transform'));
t_dc_offset = transforms.dc_offset(mnist_2,log.new_transform('Building DC offset removal transform'));
mnist_3 = t_dc_offset.code(mnist_2,log.new_transform('Removing DC component'));
t_mean_substract = transforms.mean_substract(mnist_3,log.new_transform('Building mean substract transform'));
mnist_4 = t_mean_substract.code(mnist_3,log.new_transform('Substracting mean'));
t_zca = transforms.zca(mnist_4,log.new_transform('Building ZCA transform'));
mnist_5 = t_zca.code(mnist_4,log.new_transform('Applying ZCA transform'));
mnist_6 = datasets.image.from_dataset(mnist_5,1,PATCH_ROW_COUNT,PATCH_COL_COUNT);

mnist = mnist_6;

clear mnist_raw mnist_1 mnist_2 mnist_3 mnist_4 mnist_5 mnist_6;

log.end_node();

%% Train sparse coder.

log.beg_node('Performing meta-search for best parameters');

t_sparse = cell(length(params_list),1);

for ii = 1:length(params_list)
    log.beg_node('Parameter set #%d/%d',ii,length(params_list));
    log.beg_node('Configuration');
    log.message(params.to_string(params_list(ii)));
    log.end_node();

    t_sparse{ii} = transforms.sparse.gdmp(mnist,params_list(ii).coding_fn,...
                                          params_list(ii).sparse_word_count,params_list(ii).sparse_coeffs_count,1e-3,1e-6,20,log.new_transform('Training sparse coder'));
                                      
    log.message('Final error: %.2f',t_sparse{ii}.saved_mse(end));                                      
                                      
    log.end_node();
end

log.end_node();

log.end_experiment();