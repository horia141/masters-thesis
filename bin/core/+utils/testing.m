classdef testing
    methods (Static,Access=public)        
        function [patches,t_patch,t_dc_offset,t_zca] = mnist_patches_for_dict_learning(patches_count,patch_row_count,patch_col_count,do_patch_zca)
            assert(check.scalar(patches_count));
            assert(check.natural(patches_count));
            assert(patches_count >= 1);
            assert(check.scalar(patch_row_count));
            assert(check.natural(patch_row_count));
            assert(patch_row_count >= 1);
            assert(patch_row_count <= 28);
            assert(check.scalar(patch_col_count));
            assert(check.natural(patch_col_count));
            assert(patch_col_count >= 1);
            assert(patch_col_count <= 28);
            assert(check.scalar(do_patch_zca));
            assert(check.logical(do_patch_zca));

            images = dataset.load('../../data/mnist.train.mat');
            
            t_patch = transforms.image.patch_extract(images,patches_count,patch_row_count,patch_col_count,0.01);
            
            patches_1 = t_patch.code(images);
            patches_1f = dataset.flatten_image(patches_1);
            
            t_dc_offset = transforms.record.dc_offset(patches_1f);
            
            patches_2 = t_dc_offset.code(patches_1f);
            
            if do_patch_zca
                t_zca = transforms.record.zca(patches_2);
                patches = t_zca.code(patches_2);
            else
                t_zca = {};
                patches = patches_2;
            end
        end
        
        function [patches_r,patches_g,patches_b,t_patch,t_dc_offset,t_zca] = cifar10_patches_for_dict_learning(patches_count,patch_row_count,patch_col_count,do_patch_zca)
            assert(check.scalar(patches_count));
            assert(check.natural(patches_count));
            assert(patches_count >= 1);
            assert(check.scalar(patch_row_count));
            assert(check.natural(patch_row_count));
            assert(patch_row_count >= 1);
            assert(patch_row_count <= 28);
            assert(check.scalar(patch_col_count));
            assert(check.natural(patch_col_count));
            assert(patch_col_count >= 1);
            assert(patch_col_count <= 28);
            assert(check.scalar(do_patch_zca));
            assert(check.logical(do_patch_zca));

            images = dataset.load('../../data/cifar10.train.mat');
            
            t_patch_r = transforms.image.patch_extract(images(:,:,1,:),patches_count,patch_row_count,patch_col_count,0.01);
            t_patch_g = transforms.image.patch_extract(images(:,:,2,:),patches_count,patch_row_count,patch_col_count,0.01);
            t_patch_b = transforms.image.patch_extract(images(:,:,3,:),patches_count,patch_row_count,patch_col_count,0.01);
            t_patch = {t_patch_r t_patch_g t_patch_b};
            
            patches_1_r = t_patch_r.code(images(:,:,1,:));
            patches_1_g = t_patch_g.code(images(:,:,2,:));
            patches_1_b = t_patch_b.code(images(:,:,3,:));
            patches_1f_r = dataset.flatten_image(patches_1_r);
            patches_1f_g = dataset.flatten_image(patches_1_g);
            patches_1f_b = dataset.flatten_image(patches_1_b);
            
            t_dc_offset_r = transforms.record.dc_offset(patches_1f_r);
            t_dc_offset_g = transforms.record.dc_offset(patches_1f_g);
            t_dc_offset_b = transforms.record.dc_offset(patches_1f_b);
            t_dc_offset = {t_dc_offset_r t_dc_offset_g t_dc_offset_b};
            
            patches_2_r = t_dc_offset_r.code(patches_1f_r);
            patches_2_g = t_dc_offset_g.code(patches_1f_g);
            patches_2_b = t_dc_offset_b.code(patches_1f_b);
            
            if do_patch_zca
                t_zca_r = transforms.record.zca(patches_2_r);
                t_zca_g = transforms.record.zca(patches_2_g);
                t_zca_b = transforms.record.zca(patches_2_b);
                t_zca = {t_zca_r t_zca_g t_zca_b};
                patches_r = t_zca_r.code(patches_2_r);
                patches_g = t_zca_g.code(patches_2_g);
                patches_b = t_zca_b.code(patches_2_b);
            else
                t_zca = {};
                patches_r = patches_2_r;
                patches_g = patches_2_g;
                patches_b = patches_2_b;
            end
        end
        
        function [patches,t_patch,t_dc_offset,t_zca] = smallnorb_patches_for_dict_learning(patches_count,patch_row_count,patch_col_count,do_patch_zca)
            assert(check.scalar(patches_count));
            assert(check.natural(patches_count));
            assert(patches_count >= 1);
            assert(check.scalar(patch_row_count));
            assert(check.natural(patch_row_count));
            assert(patch_row_count >= 1);
            assert(patch_row_count <= 28);
            assert(check.scalar(patch_col_count));
            assert(check.natural(patch_col_count));
            assert(patch_col_count >= 1);
            assert(patch_col_count <= 28);
            assert(check.scalar(do_patch_zca));
            assert(check.logical(do_patch_zca));

            images = dataset.load('../../data/norbsmall.train.mat');
            images = images(:,:,1,:);
            
            t_patch = transforms.image.patch_extract(images,patches_count,patch_row_count,patch_col_count,0.01);
            
            patches_1 = t_patch.code(images);
            patches_1f = dataset.flatten_image(patches_1);
            
            t_dc_offset = transforms.record.dc_offset(patches_1f);
            
            patches_2 = t_dc_offset.code(patches_1f);
            
            if do_patch_zca
                t_zca = transforms.record.zca(patches_2);
                patches = t_zca.code(patches_2);
            else
                t_zca = {};
                patches = patches_2;
            end
        end
    end
end
