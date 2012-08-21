ROW_SKIP = 4;
COL_SKIP = 4;

%load ../data/icdar/letters_nonletters_train.mat
%load ../data/icdar/letters_nonletters_test.mat
load ../data/icdar/letter_detectors.mat

test_image = imread('../data/icdar/SceneTrialTest/ryoungt_05.08.2002/aPICT0034.JPG');
test_image = imresize(rgb2gray(double(test_image) / 256),0.5);

out_image_row_count = ceil((size(test_image,1) - DESIRED_SIZES(end) + 1) / ROW_SKIP);
out_image_col_count = ceil((size(test_image,2) - DESIRED_SIZES(end) + 1) / COL_SKIP);
out_images = zeros(out_image_row_count,out_image_col_count,length(DESIRED_SIZES));

for idx = 1:length(DESIRED_SIZES)
    fprintf('Patch size %d\n',DESIRED_SIZES(idx));

    patches = zeros(DESIRED_SIZES(idx),DESIRED_SIZES(idx),1,out_image_row_count * out_image_col_count);
    current_patch = 1;

    for ii = 1:COL_SKIP:(size(test_image,2) - DESIRED_SIZES(end) + 1)
        for jj = 1:ROW_SKIP:(size(test_image,1) - DESIRED_SIZES(end) + 1)
            patches(:,:,1,current_patch) = test_image(jj:(jj + DESIRED_SIZES(idx) - 1),ii:(ii + DESIRED_SIZES(idx) - 1));
            current_patch = current_patch + 1;
        end
    end

    patches_coded = saved_codors(idx).code(patches);
    [~,patch_type] = saved_cls(idx).classify(patches_coded,-1);
    patch_type = patch_type(1,:);

    out_images(:,:,idx) = reshape(patch_type,out_image_row_count,out_image_col_count);
end

