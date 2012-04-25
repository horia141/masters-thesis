%% Setup experiment.

PATCHES_COUNT = 100000;
PATCH_ROW_COUNT = 10;
PATCH_COL_COUNT = 10;
PATCH_MIN_VARIANCE = 0.03;
SPARSE_WORD_COUNT = 100;
SPARSE_COEFFS_COUNT = 5;

hnd = logging.handlers.stdout(logging.level.All);
log = logging.logger({hnd});

%% Load data and do initial pre-processing.

scenes_raw = datasets.image.load_from_dir('../data/scenes','gray',[1600 1200],log.new_dataset_io('Loading natural scenes training data'));
t_patches = transforms.image.patch_extract(scenes_raw,PATCHES_COUNT,PATCH_ROW_COUNT,PATCH_COL_COUNT,PATCH_MIN_VARIANCE,log.new_transform('Building patch extract transform'));
scenes_1 = t_patches.code(scenes_raw,log.new_transform('Extracting patches'));
t_dc_offset = transforms.dc_offset(scenes_1,log.new_transform('Building DC offset removal transform'));
scenes_2 = t_dc_offset.code(scenes_1,log.new_transform('Removing DC component'));
t_mean_substract = transforms.mean_substract(scenes_2,log.new_transform('Building mean substract transform'));
scenes_3 = t_mean_substract.code(scenes_2,log.new_transform('Substracting mean'));
t_zca = transforms.zca(scenes_3,log.new_transform('Building ZCA transform'));
scenes_4 = t_zca.code(scenes_3,log.new_transform('Applying ZCA transform'));
scenes_5 = datasets.image.from_dataset(scenes_4,1,PATCH_ROW_COUNT,PATCH_COL_COUNT);

scenes = scenes_5;

clear scenes_raw scenes_1 scenes_2 scenes_3 scenes_4 scenes_5;

t_sparse = transforms.sparse.gdmp(scenes,@transforms.sparse.gdmp.ortho_matching_pursuit,SPARSE_WORD_COUNT,SPARSE_COEFFS_COUNT,1e-3,1e-6,20,log);
