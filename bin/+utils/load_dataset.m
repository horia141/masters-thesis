classdef load_dataset
    methods (Static,Access=public)
        function [s,s_ci] = iris()
            [s,s_ci] = utils.load_dataset.g_record_csvfile('../data/iris/iris.csv',{'%f' 4},true,',');
        end

        function [s,s_ci] = wine()
            [s,s_ci] = utils.load_dataset.g_record_csvfile('../data/wine/wine.csv',{'%f' 13},true,',');
        end

        function [s_tr,s_tr_ci,s_ts,s_ts_ci] = mnist()
            [s_tr,s_tr_ci] = utils.load_dataset.g_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte');
            [s_ts,s_ts_ci] = utils.load_dataset.g_mnist('../data/mnist/t10k-images-idx3-ubyte','../data/mnist/t10k-labels-idx1-ubyte');
        end
        
        function [s_tr,s_tr_ci,s_ts,s_ts_ci] = cifar10()
            [s_tr,s_tr_ci] = utils.load_dataset.g_cifar({'../data/cifar10/data_batch_1.mat' ...
                                                         '../data/cifar10/data_batch_2.mat' ...
                                                         '../data/cifar10/data_batch_3.mat' ...
                                                         '../data/cifar10/data_batch_4.mat' ...
                                                         '../data/cifar10/data_batch_5.mat'},...
                                                         '../data/cifar10/batches.meta.mat');
            [s_ts,s_ts_ci] = utils.load_dataset.g_cifar('../data/cifar10/test_batch.mat','../data/cifar10/batches.meta.mat');
        end

        function [s_tr,s_tr_ci,s_ts,s_ts_ci] = cifar100()
            [s_tr,s_tr_ci] = utils.load_dataset.g_cifar('../data/cifar100/train.mat','../data/cifar100/meta.mat');
            [s_ts,s_ts_ci] = utils.load_dataset.g_cifar('../data/cifar100/test.mat','../data/cifar100/meta.mat');
        end

        function [s,s_ci] = orl_faces()
            [s,s_ci] = utils.load_dataset.g_image_from_dirs('../data/orl_faces',[-1 -1]);
        end
        
        function [s_tr,s_tr_ci,s_ts,s_ts_ci] = smallnorb()
            [s_tr,s_tr_ci] = utils.load_dataset.g_norb('../data/smallnorb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','../data/smallnorb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat');
            [s_ts,s_ts_ci] = utils.load_dataset.g_norb('../data/smallnorb/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','../data/smallnorb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat');
        end

        function [s] = scenes()
            s = utils.load_dataset.g_image_from_dir('../data/scenes',[1200 1600]);
        end

        function [s,varargout] = g_record_csvfile(path,instance_format,has_classifier_info,delimiter)
            assert(check.scalar(path));
            assert(check.string(path));
            assert(check.scalar(instance_format) || check.vector(instance_format));
            assert((check.scalar(instance_format) && check.string(instance_format)) || ...
                   (check.vector(instance_format) && check.cell(instance_format) && ...
                    check.checkf(@check.scalar,instance_format(1:2:end)) && ...
                    check.checkf(@check.string,instance_format(1:2:end)) && ...
                    check.checkf(@check.scalar,instance_format(2:2:end)) && ...
                    check.checkf(@check.natural,instance_format(2:2:end)) && ...
                    check.checkf(@(ii) ii >= 1,instance_format(2:2:end))));
            assert(check.scalar(has_classifier_info));
            assert(check.logical(has_classifier_info));
            assert(check.scalar(delimiter));
            assert(check.string(delimiter));

            if check.scalar(instance_format)
                instance_format_t = {instance_format 1};
            else
                instance_format_t = instance_format;
            end
            
            if has_classifier_info
                full_format = '%s';
            else
                full_format = '';
            end
            
            for ii = 1:2:length(instance_format_t)
                full_format = sprintf('%s%s',full_format,repmat(instance_format_t{ii},1,instance_format_t{ii+1}));
            end
            
            try
                [file_fid,file_msg] = fopen(path,'rt');
                
                if file_fid == -1
                    throw(MException('master:NoLoad',...
                             sprintf('Could not load CSV file "%s": %s!',path,file_msg)));
                end
                
                sample_raw = textscan(file_fid,full_format,'delimiter',delimiter);
                
                fclose(file_fid);
            catch exp
                throw(MException('master:NoLoad',exp.message));
            end
            
            if has_classifier_info
                if ~check.checkv(check.checkf(@check.number,sample_raw(:,2:end)))
                    throw(MException('master:NoLoad',...
                             sprintf('File "%s" has an invalid format!',path)));
                end
                
                [labels_idx,labels] = grp2idx(sample_raw{:,1});
                
                s = double(cell2mat(sample_raw(:,2:end))');
                varargout{1} = classifier_info(labels,labels_idx);
            else
                if ~check.checkv(check.checkf(@check.number,sample_raw))
                    throw(MException('master:NoLoad',...
                             sprintf('File "%s" has an invalid format!',path)));
                end
                
                s = double(cell2mat(sample_raw));
            end
        end
        
        function [s,s_ci] = g_mnist(data_path,meta_path)
            assert(check.scalar(data_path));
            assert(check.string(data_path));
            assert(check.scalar(meta_path));
            assert(check.string(meta_path));

            [data_fid,data_msg] = fopen(data_path,'rb');
            
            if data_fid == -1
                throw(MException('master:NoLoad',...
                         sprintf('Could not load images in "%s": %s!',data_path,data_msg)))
            end
            
            [meta_fid,meta_msg] = fopen(meta_path,'rb');
            
            if meta_fid == -1
                fclose(data_fid);
                throw(MException('master:NoLoad',...
                         sprintf('Could not load labels in "%s": %s!',meta_path,meta_msg)))
            end
            
            try
                data_magic = utils.load_dataset.high2low(fread(data_fid,4,'uint8=>uint32'));
                
                if data_magic ~= 2051
                    throw(MException('master:NoLoad',...
                             sprintf('Images file "%s" not in MNIST format!',data_path)));
                end
                
                meta_magic = utils.load_dataset.high2low(fread(meta_fid,4,'uint8=>uint32'));
                
                if meta_magic ~= 2049
                    throw(MException('master:NoLoad',...
                             sprintf('Labels file "%s" not in MNIST format!',meta_path)));
                end
                
                data_count = utils.load_dataset.high2low(fread(data_fid,4,'uint8=>uint32'));
                layer_count = 1;
                row_count = utils.load_dataset.high2low(fread(data_fid,4,'uint8=>uint32'));
                col_count = utils.load_dataset.high2low(fread(data_fid,4,'uint8=>uint32'));
                
                meta_count = utils.load_dataset.high2low(fread(meta_fid,4,'uint8=>uint32'));
                
                if data_count ~= meta_count
                    throw(MException('master:NoLoad',...
                             sprintf('Different number of labels in "%s" for images in "%s"!',meta_path,data_path)));
                end
                
                images_t1 = fread(data_fid,row_count * col_count * 1 * data_count,'uint8=>double') ./ 255;
                images_t2 = reshape(images_t1,row_count,col_count,1,data_count);
                images = permute(images_t2,[2 1 3 4]);

                labels = fread(meta_fid,[data_count 1],'uint8');
                
                fclose(data_fid);
                fclose(meta_fid);
            catch exp
                fclose(data_fid);
                fclose(meta_fid);
                throw(MException('master:NoLoad',exp.message));
            end
            
            s = images;
            s_ci = classifier_info({'d0' 'd1' 'd2' 'd3' 'd4' 'd5' 'd6' 'd7' 'd8' 'd9'},labels + 1);
        end
        
        function [s,s_ci] = g_cifar(data_paths,meta_path)
            assert((check.scalar(data_paths) && check.string(data_paths)) || ...
                   (check.vector(data_paths) && check.cell(data_paths) && check.checkf(@check.scalar,data_paths) && check.checkf(@check.string,data_paths)));
            assert(check.scalar(meta_path));
            assert(check.string(meta_path));
            
            if check.scalar(data_paths)
                data_paths_t = {data_paths};
            else
                data_paths_t = data_paths;
            end
            
            try
                meta_info = load(meta_path);
                data_info = cell(1,length(data_paths_t));
                
                for ii = 1:length(data_paths_t)
                    data_info{ii} = load(data_paths_t{ii});
                end
                
                indices = [0 cumsum(cellfun(@(c)size(c.data,1),data_info))];
                s = zeros(32,32,3,sum(cellfun(@(c)size(c.data,1),data_info)));
                s_ci = classifier_info(meta_info.label_names,1 + cell2mat(cellfun(@(c)double(c.labels)',data_info,'UniformOutput',false)));
                
                for ii = 1:length(data_paths_t)
                    idx1 = indices(ii) + 1;
                    idx2 = indices(ii + 1);
                    local_data = double(data_info{ii}.data') / 255;
                    
                    s(:,:,:,idx1:idx2) = reshape(local_data,32,32,3,idx2 - idx1 + 1);
                end
                
                s = permute(s,[2 1 3 4]);
            catch exp
                throw(MException('master:NoLoad',...
                         sprintf('Could not load images: %s',exp.message)));
            end
        end
        
        function [s,s_ci] = g_norb(data_path,meta_path)
            assert(check.scalar(data_path));
            assert(check.string(data_path));
            assert(check.scalar(meta_path));
            assert(check.string(meta_path));
            
            data_fid = fopen(data_path,'rb');
            
            if data_fid == -1
                throw(MException('master:NoLoad',...
                         sprintf('Could not load images in "%s": %s!',data_path,data_msg)))
            end
            
            [meta_fid,meta_msg] = fopen(meta_path,'rb');
            
            if meta_fid == -1
                fclose(data_fid);
                throw(MException('master:NoLoad',...
                         sprintf('Could not load labels in "%s": %s!',meta_path,meta_msg)))
            end
            
            try
                data_magic = fread(data_fid,1,'uint32');
                
                if ~check.same(data_magic,507333717)
                    throw(MException('master:NoLoad',...
                             sprintf('Images file "%s" not in NORB format!',data_path)));
                end
                
                meta_magic = fread(meta_fid,1,'uint32');
                
                if ~check.same(meta_magic,507333716)
                    throw(MException('master:NoLoad',...
                             sprintf('Labels file "%s" not in NORB format!',meta_path)));
                end
                
                data_dims_count = fread(data_fid,1,'uint32');
                
                if data_dims_count ~= 4
                    throw(MException('master:NoLoad',...
                             sprintf('Image file "%s" not in NORB format!',data_path)));
                end
                
                data_count = fread(data_fid,1,'uint32=>uint64');
                layer_count = fread(data_fid,1,'uint32=>uint64');
                col_count = fread(data_fid,1,'uint32=>uint64');
                row_count = fread(data_fid,1,'uint32=>uint64');
                
                meta_dims_count = fread(meta_fid,1,'uint32');
                
                if meta_dims_count ~= 1
                    throw(MException('master:NoLoad',...
                             sprintf('Labels file "%s" not in NORB format!',meta_path)));
                end
                
                meta_count = fread(meta_fid,1,'uint32=>uint64');
                ignore1 = fread(meta_fid,1,'uint32');
                ignore2 = fread(meta_fid,1,'uint32');
                
                if data_count ~= meta_count
                    throw(MException('master:NoLoad',...
                             sprintf('Different number of labels in "%s" for images in "%s"!',meta_path,data_path)));
                end
                
                if ignore1 ~= 1 || ignore2 ~= 1
                    throw(MException('master:NoLoad',...
                             sprintf('Labels file "%s" not in NORB format!',meta_path)));
                end

                images_t1 = fread(data_fid,row_count*col_count*layer_count*data_count,'uint8=>double') ./ 255;
                images_t2 = reshape(images_t1,row_count,col_count,layer_count,data_count);
                images = permute(images_t2,[2 1 3 4]);
                
                labels = fread(meta_fid,meta_count,'uint32');
                
                fclose(meta_fid);
                fclose(data_fid);                
            catch exp
                fclose(data_fid);
                fclose(meta_fid);
                throw(MException('master:NoLoad',exp.message));
            end
            
            s = images;
            s_ci = classifier_info({'animals' 'humans' 'airplanes' 'trucks' 'cars'},labels + 1);
        end
        
        function [s] = g_image_from_dir(path,force_size)
            assert(check.scalar(path));
            assert(check.string(path));
            assert(check.vector(force_size));
            assert(length(force_size) == 2);
            assert((check.natural(force_size) && check.checkv(force_size >= 1)) || ...
                   check.checkv(force_size == -1));

            paths = dir(path);
            paths = paths(3:end);
            images = [];
            current_image = 1;
            
            for ii = 1:length(paths)
                try
                    image = double(imread(fullfile(path,paths(ii).name))) / 255;
                    
                    if check.checkv(force_size ~= [-1 -1])
                        image = imresize(image,force_size);
                        image = utils.common.clamp_images_to_unit(image);
                    end
                    
                    if (current_image > 1) && ...
                        (~check.checkv(size(image) == size(images(:,:,:,1))))
                        throw(MException('master:NoLoad',...
                                         'Images are of different geometries!'));
                    end

                    images = cat(4,images,image);
                    current_image = current_image + 1;
                catch exp
                    if isempty(regexp(exp.identifier,'MATLAB:(.*:)?imread:.*','ONCE'))
                        throw(MException('master:NoLoad',exp.message));
                    end
                end
            end
            
            if isempty(images)
                throw(MException('master:NoLoad',...
                                 'Could not find any acceptable images in the directory.'));
            end
            
            s = images;
        end
        
        function [s,s_ci] = g_image_from_dirs(path,force_size)
            assert(check.scalar(path));
            assert(check.string(path));
            assert(check.vector(force_size));
            assert(length(force_size) == 2);
            assert((check.natural(force_size) && check.checkv(force_size >= 1)) || ...
                   check.checkv(force_size == -1));

            paths = dir(path);
            paths = paths(3:end);
            images = [];
            labels = {};
            labels_idx = [];
            
            current_class = 1;
            
            for ii = 1:length(paths)
                if paths(ii).isdir
                    local_images = utils.load_dataset.g_image_from_dir(fullfile(path,paths(ii).name),force_size);
                    images = cat(4,images,local_images);
                    labels = cat(2,labels,paths(ii).name);
                    labels_idx = cat(2,labels_idx,current_class * ones(1,dataset.count(local_images)));
                    current_class = current_class + 1;
                end
            end
            
            s = images;
            s_ci = classifier_info(labels,labels_idx);
        end
    end
    
    methods (Static,Access=private)
        function [out] = high2low(bytes)
            out = bitshift(bytes(4),0) + bitshift(bytes(3),8) + ...
                  bitshift(bytes(2),16) + bitshift(bytes(1),24);
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "utils.load_dataset".\n');
            
            fprintf('  Function "iris".\n');
            
            [s,s_ci] = utils.load_dataset.iris();
            
            assert(check.checkv(size(s) == [4 150]));
            assert(check.number(s));
            assert(check.same(s_ci.labels,{'Iris-setosa' 'Iris-versicolor' 'Iris-virginica'}));
            assert(s_ci.labels_count == 3);
            assert(s_ci.compatible(s));
            
            clearvars -except test_figure;
            
            fprintf('  Function "wine".\n');
            
            [s,s_ci] = utils.load_dataset.wine();
            
            assert(check.checkv(size(s) == [13 178]));
            assert(check.number(s));
            assert(check.same(s_ci.labels,{'1' '2' '3'}));
            assert(s_ci.labels_count == 3);
            assert(s_ci.compatible(s));
            
            clearvars -except test_figure;
            
            fprintf('  Function "mnist".\n');
            
            [s_tr,s_tr_ci,s_ts,s_ts_ci] = utils.load_dataset.mnist();
            
            assert(check.checkv(size(s_tr) == [28 28 1 60000]));
            assert(check.unitreal(s_tr));
            assert(check.same(s_tr_ci.labels,{'d0' 'd1' 'd2' 'd3' 'd4' 'd5' 'd6' 'd7' 'd8' 'd9'}));
            assert(s_tr_ci.labels_count == 10);
            assert(s_tr_ci.compatible(s_tr));
            assert(check.checkv(size(s_ts) == [28 28 1 10000]));
            assert(check.unitreal(s_ts));
            assert(check.same(s_ts_ci.labels,{'d0' 'd1' 'd2' 'd3' 'd4' 'd5' 'd6' 'd7' 'd8' 'd9'}));
            assert(s_ts_ci.labels_count == 10);
            assert(s_ts_ci.compatible(s_ts));
            
            clearvars -except test_figure;
            
            fprintf('  Function "cifar10".\n');
            
            [s_tr,s_tr_ci,s_ts,s_ts_ci] = utils.load_dataset.cifar10();
            
            assert(check.checkv(size(s_tr) == [32 32 3 50000]));
            assert(check.unitreal(s_tr));
            assert(check.same(s_tr_ci.labels,{'airplane' 'automobile' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck'}));
            assert(s_tr_ci.labels_count == 10);
            assert(s_tr_ci.compatible(s_tr));
            assert(check.checkv(size(s_ts) == [32 32 3 10000]));
            assert(check.unitreal(s_ts));
            assert(check.same(s_ts_ci.labels,{'airplane' 'automobile' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck'}));
            assert(s_ts_ci.labels_count == 10);
            assert(s_ts_ci.compatible(s_ts));
            
            clearvars -except test_figure;
            
            fprintf('  Function "cifar100".\n');
            
            [s_tr,s_tr_ci,s_ts,s_ts_ci] = utils.load_dataset.cifar100();
            
            assert(check.checkv(size(s_tr) == [32 32 3 50000]));
            assert(check.unitreal(s_tr));
            assert(check.same(s_tr_ci.labels,{'apple' 'aquarium_fish' 'baby' 'bear' 'beaver' 'bed' 'bee' 'beetle' 'bicycle' 'bottle' ...
                                              'bowl' 'boy' 'bridge' 'bus' 'butterfly' 'camel' 'can' 'castle' 'caterpillar' 'cattle' 'chair' ...
                                              'chimpanzee' 'clock' 'cloud' 'cockroach' 'couch' 'crab' 'crocodile' 'cup' 'dinosaur' 'dolphin' ...
                                              'elephant' 'flatfish' 'forest' 'fox' 'girl' 'hamster' 'house' 'kangaroo' 'keyboard' 'lamp' ...
                                              'lawn_mower' 'leopard' 'lion' 'lizard' 'lobster' 'man' 'maple_tree' 'motorcycle' 'mountain' ...
                                              'mouse' 'mushroom' 'oak_tree' 'orange' 'orchid' 'otter' 'palm_tree' 'pear' 'pickup_truck' ...
                                              'pine_tree' 'plain' 'plate' 'poppy' 'porcupine' 'possum' 'rabbit' 'raccoon' 'ray' 'road' ...
                                              'rocket' 'rose' 'sea' 'seal' 'shark' 'shrew' 'skunk' 'skyscraper' 'snail' 'snake' 'spider' ...
                                              'squirrel' 'streetcar' 'sunflower' 'sweet_pepper' 'table' 'tank' 'telephone' 'television' ...
                                              'tiger' 'tractor' 'train' 'trout' 'tulip' 'turtle' 'wardrobe' 'whale' 'willow_tree' 'wolf' ...
                                              'woman' 'worm'}));
            assert(s_tr_ci.labels_count == 100);
            assert(s_tr_ci.compatible(s_tr));
            assert(check.checkv(size(s_ts) == [32 32 3 10000]));
            assert(check.unitreal(s_ts));
            assert(check.same(s_ts_ci.labels,{'apple' 'aquarium_fish' 'baby' 'bear' 'beaver' 'bed' 'bee' 'beetle' 'bicycle' 'bottle' ...
                                              'bowl' 'boy' 'bridge' 'bus' 'butterfly' 'camel' 'can' 'castle' 'caterpillar' 'cattle' 'chair' ...
                                              'chimpanzee' 'clock' 'cloud' 'cockroach' 'couch' 'crab' 'crocodile' 'cup' 'dinosaur' 'dolphin' ...
                                              'elephant' 'flatfish' 'forest' 'fox' 'girl' 'hamster' 'house' 'kangaroo' 'keyboard' 'lamp' ...
                                              'lawn_mower' 'leopard' 'lion' 'lizard' 'lobster' 'man' 'maple_tree' 'motorcycle' 'mountain' ...
                                              'mouse' 'mushroom' 'oak_tree' 'orange' 'orchid' 'otter' 'palm_tree' 'pear' 'pickup_truck' ...
                                              'pine_tree' 'plain' 'plate' 'poppy' 'porcupine' 'possum' 'rabbit' 'raccoon' 'ray' 'road' ...
                                              'rocket' 'rose' 'sea' 'seal' 'shark' 'shrew' 'skunk' 'skyscraper' 'snail' 'snake' 'spider' ...
                                              'squirrel' 'streetcar' 'sunflower' 'sweet_pepper' 'table' 'tank' 'telephone' 'television' ...
                                              'tiger' 'tractor' 'train' 'trout' 'tulip' 'turtle' 'wardrobe' 'whale' 'willow_tree' 'wolf' ...
                                              'woman' 'worm'}));
            assert(s_ts_ci.labels_count == 100);
            assert(s_ts_ci.compatible(s_ts));
            
            clearvars -except test_figure;
            
            fprintf('  Function "smallnorb".\n');
            
            % do all tests here as well. For the moment, this is too big
            % for testing on smaller machines.
            
            fprintf('  Function "orl_faces".\n');
            
            [s,s_ci] = utils.load_dataset.orl_faces();
            
            assert(check.checkv(size(s) == [112 92 1 400]));
            assert(check.unitreal(s));
            assert(check.same(s_ci.labels,{'s1' 's10' 's11' 's12' 's13' 's14' 's15' 's16' 's17' 's18' 's19' 's2' 's20' ...
                                           's21' 's22' 's23' 's24' 's25' 's26' 's27' 's28' 's29' 's3' 's30' 's31' 's32' ...
                                           's33' 's34' 's35' 's36' 's37' 's38' 's39' 's4' 's40' 's5' 's6' 's7' 's8' 's9'}));
            assert(s_ci.labels_count == 40);
            assert(s_ci.compatible(s));
            
            clearvars -except test_figure;
            
            fprintf('  Function "scenes".\n');
            
            s = utils.load_dataset.scenes();
            
            assert(check.checkv(size(s) == [1200 1600 3 9]));
            assert(check.unitreal(s));

            clearvars -except test_figure;
            
            fprintf('  Function "g_record_csvfile".\n');
            
            fprintf('    With Iris data and simple format.\n');
            
            [s,s_ci] = utils.load_dataset.g_record_csvfile('../test/iris/iris.csv','%f%f%f%f',true,',');
            
            assert(check.checkv(size(s) == [4 150]));
            assert(check.number(s));
            assert(check.same(s_ci.labels,{'Iris-setosa' 'Iris-versicolor' 'Iris-virginica'}));
            assert(s_ci.labels_count == 3);
            assert(s_ci.compatible(s));
            
            clearvars -except test_figure;
            
            fprintf('    With Iris data and complex format.\n');
            
            [s,s_ci] = utils.load_dataset.g_record_csvfile('../test/iris/iris.csv',{'%f' 2 '%f' 2},true,',');
            
            assert(check.checkv(size(s) == [4 150]));
            assert(check.number(s));
            assert(check.same(s_ci.labels,{'Iris-setosa' 'Iris-versicolor' 'Iris-virginica'}));
            assert(s_ci.labels_count == 3);
            assert(s_ci.compatible(s));
            
            clearvars -except test_figure;
            
            fprintf('    With invalid external inputs.\n');
            
            try
                utils.load_dataset.g_record_csvfile('../test/wine/wine.csv',{'%s' 2 '%f' 11},true,',');
                assert(false);
            catch exp
                if strcmp(exp.message,'File "../test/wine/wine.csv" has an invalid format!')
                    fprintf('      Passes "Invalid format!" test.\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except test_figure;
            
            fprintf('  Function "g_mnist".\n');
            
            fprintf('    With MNIST test data.\n');
            
            [s,s_ci] = utils.load_dataset.g_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte');
            
            assert(check.checkv(size(s) == [28 28 1 10000]));
            assert(check.unitreal(s));
            assert(check.same(s_ci.labels,{'d0' 'd1' 'd2' 'd3' 'd4' 'd5' 'd6' 'd7' 'd8' 'd9'}));
            assert(s_ci.labels_count == 10);
            assert(s_ci.compatible(s));
            
            clearvars -except test_figure;
            
            fprintf('    With invalid external inputs.\n');

            try
                utils.load_dataset.g_mnist('../test/scenes_small/scenes_small1.jpg','../test/mnist/t10k-labels-idx1-ubyte');
            catch exp
                if strcmp(exp.message,'Images file "../test/scenes_small/scenes_small1.jpg" not in MNIST format!')
                    fprintf('      Passes "Not in MNIST format!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                utils.load_dataset.g_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/scenes_small/scenes_small1.jpg');
            catch exp
                if strcmp(exp.message,'Labels file "../test/scenes_small/scenes_small1.jpg" not in MNIST format!')
                    fprintf('      Passes "Not in MNIST format!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                utils.load_dataset.g_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte');
            catch exp
                if strcmp(exp.message,'Different number of labels in "../test/mnist/t10k-labels-idx1-ubyte" for images in "../test/mnist/t10k-images-idx3-ubyte"!')
                    fprintf('      Passes "Different number of labels!".\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except test_figure;
            
            fprintf('  Function "g_cifar".\n');
            
            fprintf('    With CIFAR10 test data.\n');
            
            [s,s_ci] = utils.load_dataset.g_cifar('../test/cifar/test.mat','../test/cifar/meta.mat');
            
            assert(check.checkv(size(s) == [32 32 3 10000]));
            assert(check.unitreal(s));
            assert(check.same(s_ci.labels,{'airplane' 'automobile' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'ship' 'truck'}));
            assert(s_ci.labels_count == 10);
            assert(s_ci.compatible(s));
            
            clearvars -except test_figure;
            
            fprintf('    With invalid external inputs.\n');
            
            % test for data file not with required fields
            % test for data stores in data file not good (various size
            % mismatches, for example);
            % test for meta file not with required fields
            % test for data stored in meta file not good.
            % test for incompatibilities between data file and meta file.
            % test for incompatibilities between two data files.
            
            clearvars -except test_figure;
            
            fprintf('  Function "g_norb".\n');
            
            % do all tests here as well. For the moment, this is too big
            % for testing on smaller machines.
            
            fprintf('  Function "g_image_from_dir".\n');
            
            fprintf('  Function "g_image_from_dirs".\n');
        end
    end
end
