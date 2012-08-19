classdef testing
    methods (Static,Access=public)
        function [images] = scenes_small()
            images = zeros(192,256,3,7);
            images(:,:,:,1) = double(imread('../test/scenes_small/scenes_small1.jpg')) / 255;
            images(:,:,:,2) = double(imread('../test/scenes_small/scenes_small2.jpg')) / 255;
            images(:,:,:,3) = double(imread('../test/scenes_small/scenes_small3.jpg')) / 255;
            images(:,:,:,4) = double(imread('../test/scenes_small/scenes_small4.jpg')) / 255;
            images(:,:,:,5) = double(imread('../test/scenes_small/scenes_small5.jpg')) / 255;
            images(:,:,:,6) = double(imread('../test/scenes_small/scenes_small6.jpg')) / 255;
            images(:,:,:,7) = double(imread('../test/scenes_small/scenes_small7.jpg')) / 255;
        end
        
        function [dict] = sparse_dictionary()
            dict_struct = load('../test/sparse_dictionary.mat');
            dict = dict_struct.saved_dict_11x11_144;
        end
        
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

            images = utils.load_dataset.g_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte');
            
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

            images = utils.load_dataset.cifar10();
            
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

            images = utils.load_dataset.smallnorb();
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
        
        function [images] = mnist()
            images = utils.load_dataset.g_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte');
        end

        function [s] = correlated_cloud()
            s = mvnrnd([3 3],[1 0.6; 0.6 0.5],10000)';
        end
        
        function [s] = two_component_cloud()
            s = [mvnrnd([0 0],[3 0; 0 0.01],200); ...
                 mvnrnd([0 0],[0.01 0; 0 3],200)]';
        end
        
        function [s] = three_component_cloud()
            s = [mvnrnd([0 0],[3 0; 0 0.01],200); ...
                 mvnrnd([0 0],[0.01 0; 0 3],200); ...
                 mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
        end

        function [s,ci] = classifier_data_3()
            s = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([3 3],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)]';
            s = sparse(s);
            ci = classifier_info({'1' '2' '3'},[1*ones(1,100) 2*ones(1,100) 3*ones(1,100)]);
        end

        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_clear_data_3()
            s = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([3 3],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)]';
            ci = classifier_info({'1' '2' '3'},[1*ones(1,100) 2*ones(1,100) 3*ones(1,100)]);
            [tr_i,ts_i] = ci.partition(0.2);
            s_tr = sparse(dataset.subsample(s,tr_i));
            s_ts = sparse(dataset.subsample(s,ts_i));
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_mostly_clear_data_3()
            s_tr = [mvnrnd([3 1],[0.01 0; 0 0.01],80);
                    mvnrnd([3 3],[0.01 0; 0 0.01],80);
                    mvnrnd([1 3],[0.01 0; 0 0.01],80)]';
            s_tr = sparse(s_tr);
            s_ts = [mvnrnd([3 1],[0.01 0; 0 0.01],18);
                    3 3;
                    1 3;
                    mvnrnd([3 3],[0.01 0; 0 0.01],18);
                    3 1;
                    1 3;
                    mvnrnd([1 3],[0.01 0; 0 0.01],18)
                    3 1;
                    3 3]';
            s_ts = sparse(s_ts);
            ci_tr = classifier_info({'1' '2' '3'},[1*ones(1,80) 2*ones(1,80) 3*ones(1,80)]);
            ci_ts = classifier_info({'1' '2' '3'},[1*ones(1,20) 2*ones(1,20) 3*ones(1,20)]);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_unclear_data_3()
            s = [mvnrnd([3 1],[0.5 0; 0 0.5],100);
                 mvnrnd([3 3],[0.5 0; 0 0.5],100);
                 mvnrnd([1 3],[0.5 0; 0 0.5],100)]';
            ci = classifier_info({'1' '2' '3'},[1*ones(1,100) 2*ones(1,100) 3*ones(1,100)]);
            [tr_i,ts_i] = ci.partition(0.2);
            s_tr = sparse(dataset.subsample(s,tr_i));
            s_ts = sparse(dataset.subsample(s,ts_i));
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end
        
        function [s,ci] = classifier_data_2()
            s = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)]';
            s = sparse(s);
            ci = classifier_info({'1' '2'},[1*ones(1,100) 2*ones(1,100)]);
        end

        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_clear_data_2()
            s = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)]';
            ci = classifier_info({'1' '2'},[1*ones(1,100) 2*ones(1,100)]);
            [tr_i,ts_i] = ci.partition(0.2);
            s_tr = sparse(dataset.subsample(s,tr_i));
            s_ts = sparse(dataset.subsample(s,ts_i));
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_mostly_clear_data_2()
            s_tr = [mvnrnd([3 1],[0.01 0; 0 0.01],80);
                    mvnrnd([1 3],[0.01 0; 0 0.01],80)]';
            s_tr = sparse(s_tr);
            s_ts = [mvnrnd([3 1],[0.01 0; 0 0.01],19);
                    1 3;
                    mvnrnd([1 3],[0.01 0; 0 0.01],19)
                    3 1]';
            s_ts = sparse(s_ts);
            ci_tr = classifier_info({'1' '2'},[1*ones(1,80) 2*ones(1,80)]);
            ci_ts = classifier_info({'1' '2'},[1*ones(1,20) 2*ones(1,20)]);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_unclear_data_2()
            s = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)]';
            ci = classifier_info({'1' '2'},[1*ones(1,100) 2*ones(1,100)]);
            [tr_i,ts_i] = ci.partition(0.2);
            s_tr = sparse(dataset.subsample(s,tr_i));
            s_ts = sparse(dataset.subsample(s,ts_i));
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "utils.testing".\n');
        end
    end
end
