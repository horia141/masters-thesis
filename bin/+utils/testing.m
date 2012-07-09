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
        
        function [images] = mnist()
            hnd = logging.handlers.zero(logging.level.Experiment);
            logg = logging.logger({hnd});
            
            images = utils.load_dataset.g_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte',logg);
            
            logg.close();
            hnd.close();
        end

        function [s] = correlated_cloud()
            s = mvnrnd([3 3],[1 0.6; 0.6 0.5],10000)';
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
            ci = classifier_info({'1' '2' '3'},[1*ones(1,100) 2*ones(1,100) 3*ones(1,100)]);
        end

        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_clear_data_3()
            s = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([3 3],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)]';
            ci = classifier_info({'1' '2' '3'},[1*ones(1,100) 2*ones(1,100) 3*ones(1,100)]);
            [tr_i,ts_i] = ci.partition(0.2);
            s_tr = dataset.subsample(s,tr_i);
            s_ts = dataset.subsample(s,ts_i);
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_mostly_clear_data_3()
            s_tr = [mvnrnd([3 1],[0.01 0; 0 0.01],80);
                    mvnrnd([3 3],[0.01 0; 0 0.01],80);
                    mvnrnd([1 3],[0.01 0; 0 0.01],80)]';
            s_ts = [mvnrnd([3 1],[0.01 0; 0 0.01],18);
                    3 3;
                    1 3;
                    mvnrnd([3 3],[0.01 0; 0 0.01],18);
                    3 1;
                    1 3;
                    mvnrnd([1 3],[0.01 0; 0 0.01],18)
                    3 1;
                    3 3]';
            ci_tr = classifier_info({'1' '2' '3'},[1*ones(1,80) 2*ones(1,80) 3*ones(1,80)]);
            ci_ts = classifier_info({'1' '2' '3'},[1*ones(1,20) 2*ones(1,20) 3*ones(1,20)]);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_unclear_data_3()
            s = [mvnrnd([3 1],[0.5 0; 0 0.5],100);
                 mvnrnd([3 3],[0.5 0; 0 0.5],100);
                 mvnrnd([1 3],[0.5 0; 0 0.5],100)]';
            ci = classifier_info({'1' '2' '3'},[1*ones(1,100) 2*ones(1,100) 3*ones(1,100)]);
            [tr_i,ts_i] = ci.partition(0.2);
            s_tr = dataset.subsample(s,tr_i);
            s_ts = dataset.subsample(s,ts_i);
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end
        
        function [s,ci] = classifier_data_2()
            s = [mvnrnd([3 1],[0.1 0; 0 0.1],100);
                 mvnrnd([1 3],[0.1 0; 0 0.1],100)]';
            ci = classifier_info({'1' '2'},[1*ones(1,100) 2*ones(1,100)]);
        end

        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_clear_data_2()
            s = [mvnrnd([3 1],[0.01 0; 0 0.01],100);
                 mvnrnd([1 3],[0.01 0; 0 0.01],100)]';
            ci = classifier_info({'1' '2'},[1*ones(1,100) 2*ones(1,100)]);
            [tr_i,ts_i] = ci.partition(0.2);
            s_tr = dataset.subsample(s,tr_i);
            s_ts = dataset.subsample(s,ts_i);
            ci_tr = ci.subsample(tr_i);
            ci_ts = ci.subsample(ts_i);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_mostly_clear_data_2()
            s_tr = [mvnrnd([3 1],[0.01 0; 0 0.01],80);
                    mvnrnd([1 3],[0.01 0; 0 0.01],80)]';
            s_ts = [mvnrnd([3 1],[0.01 0; 0 0.01],19);
                    1 3;
                    mvnrnd([1 3],[0.01 0; 0 0.01],19)
                    3 1]';
            ci_tr = classifier_info({'1' '2'},[1*ones(1,80) 2*ones(1,80)]);
            ci_ts = classifier_info({'1' '2'},[1*ones(1,20) 2*ones(1,20)]);
        end
        
        function [s_tr,s_ts,ci_tr,ci_ts] = classifier_unclear_data_2()
            s = [mvnrnd([3 1],[1 0; 0 1],100);
                 mvnrnd([1 3],[1 0; 0 1],100)]';
            ci = classifier_info({'1' '2'},[1*ones(1,100) 2*ones(1,100)]);
            [tr_i,ts_i] = ci.partition(0.2);
            s_tr = dataset.subsample(s,tr_i);
            s_ts = dataset.subsample(s,ts_i);
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
