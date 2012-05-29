classdef dataset
    methods (Static,Access=public)
        function [sample_count] = count(dataset)
            assert(tc.dataset(dataset));
            
            if tc.dataset_record(dataset)
                sample_count = size(dataset,2);
            elseif tc.dataset_image(dataset)
                sample_count = size(dataset,4);
            else
                assert(false);
            end
        end
        
        function [varargout] = geometry(dataset)
            assert(tc.dataset(dataset));

            if tc.dataset_record(dataset)
                varargout{1} = size(dataset,1);
            elseif tc.dataset_image(dataset)
                features_count = size(dataset,1) * size(dataset,2) * size(dataset,3);

                if nargout == 1
                    varargout{1} = [features_count size(dataset,1) size(dataset,2) size(dataset,3)];
                elseif nargout >= 2
                    varargout{1} = features_count;
                    varargout{2} = size(dataset,1);
                    varargout{3} = size(dataset,2);
                    varargout{4} = size(dataset,3);
                end
            end
        end
        
        function [o] = geom_compatible(geom_1,geom_2)
            assert(tc.vector(geom_1));
            assert((length(geom_1) == 1) || (length(geom_1) == 4));
            assert(tc.natural(geom_1));
            assert(tc.check(geom_1 >= 1));
            assert(tc.vector(geom_2));
            assert((length(geom_2) == 1) || (length(geom_2) == 4));
            assert(tc.natural(geom_2));
            assert(tc.check(geom_2 >= 1));
            
            o = tc.same(geom_1,geom_2);
        end

        function [sample,class_info] = load_record_csvfile(csvfile_path,data_format,delimiter,logger)
            assert(tc.scalar(csvfile_path));
            assert(tc.string(csvfile_path));
            assert(tc.scalar(data_format));
            assert(tc.string(data_format));
            assert(~exist('delimiter','var') || tc.scalar(delimiter));
            assert(~exist('delimiter','var') || tc.string(delimiter));
            assert(~exist('logger','var') || tc.scalar(logger));
            assert(~exist('logger','var') || tc.logging_logger(logger));
            assert(~exist('logger','var') || logger.active);
            
            if ~exist('delimiter','var')
                delimiter = ',';
            end
            
            if ~exist('logger','var')
                hnd_zero = logging.handlers.zero(logging.level.All);
                logger = logging.logger({hnd_zero});
            end
            
            try
                logger.message('Opening csv file "%s".',csvfile_path);
                
                [csvfile_fid,csvfile_msg] = fopen(csvfile_path,'rt');
                
                if csvfile_fid == -1
                    throw(MException('master:NoLoad',...
                             sprintf('Could not load csv file "%s": %s!',csvfile_path,csvfile_msg)))
                end
                
                logger.message('Bulk reading of CSV data.');

                sample_raw = textscan(csvfile_fid,strcat('%s',data_format),'delimiter',delimiter);

                fclose(csvfile_fid);
            catch exp
                throw(MException('master:NoLoad',exp.message));
            end
            
            if ~tc.check(tc.checkf(@tc.number,sample_raw(:,2:end)))
                throw(MException('master:InvalidFormat',...
                         sprintf('File "%s" has an invalid format!',csvfile_path)));
            end
            
            logger.message('Building dataset and labels information.');
            
            [labels_idx,labels] = grp2idx(sample_raw{:,1});
            
            sample = cell2mat(sample_raw(:,2:end))';
            class_info = classification_info(labels,labels_idx);
        end

        function [sample] = load_image_from_dir(images_dir_path,mode,force_size,logger)
            assert(tc.scalar(images_dir_path));
            assert(tc.string(images_dir_path));
            assert(~exist('mode','var') || tc.scalar(mode));
            assert(~exist('mode','var') || tc.string(mode));
            assert(~exist('mode','var') || tc.one_of(mode,'gray','original'));
            assert(~exist('force_size','var') || tc.vector(force_size));
            assert(~exist('force_size','var') || (length(force_size) == 2));
            assert(~exist('force_size','var') || tc.integer(force_size));
            assert(~exist('force_size','var') || (tc.check(force_size >= 1) || tc.check(force_size == -1)));
            assert(~exist('logger','var') || tc.scalar(logger));
            assert(~exist('logger','var') || tc.logging_logger(logger));
            assert(~exist('logger','var') || logger.active);

            if ~exist('mode','var')
                mode = 'gray';
            end
            
            if ~exist('force_size','var')
                force_size = [-1 -1];
            end
            
            if ~exist('logger','var')
                hnd_zero = logging.handlers.zero(logging.level.All);
                logger = logging.logger({hnd_zero});
            end
            
            logger.message('Listing images directory "%s".',images_dir_path);
               
            paths = dir(images_dir_path);
            images = [];
            current_image = 1;
            
            logger.beg_node('Starting reading of images');
            
            for ii = 1:length(paths)
                try
                    logger.beg_node('Reading image in "%s"',fullfile(images_dir_path,paths(ii).name));
                    
                    image = imread(fullfile(images_dir_path,paths(ii).name));
                    
                    if strcmp(mode,'gray')
                        image = double(rgb2gray(image)) ./ 255;
                    else
                        image = double(image) ./ 255;
                    end
                    
                    logger.message('Row count: %d',size(image,1));
                    logger.message('Col count: %d',size(image,2));
                    
                    if tc.check(force_size ~= [-1 -1])
                        logger.message('Resizing to %dx%d.',force_size(1),force_size(2));
                        
                        image = imresize(image,force_size);
                        
                        % Correct small domain overflows caused by resizing.
                        image = utils.clamp_images_to_unit(image);
                    end
                    
                    if (current_image > 1) && ...
                        (~tc.check(size(image) == size(images(:,:,:,1))))
                        throw(MException('master:NoLoad',...
                                         'Images are of different sizes!'));
                    end

                    images(:,:,:,current_image) = image;
                    current_image = current_image + 1;
                    
                    logger.end_node();
                catch exp
                    logger.message('Not an image or corrupted.');

                    logger.end_node();

                    if isempty(regexp(exp.identifier,'MATLAB:(.*:)?imread:.*','ONCE'))
                        throw(MException('master:NoLoad',exp.message));
                    end
                end
            end
            
            logger.end_node();
            
            if isempty(images)
                throw(MException('master:NoLoad',...
                                 'Could not find any acceptable images in the directory.'));
            end
            
            logger.message('Building dataset.');
            
            sample = images;
        end

        function [sample,class_info] = load_image_mnist(images_path,labels_path,logger)
            assert(tc.scalar(images_path));
            assert(tc.string(images_path));
            assert(tc.scalar(labels_path));
            assert(tc.string(labels_path));
            assert(~exist('logger','var') || tc.scalar(logger));
            assert(~exist('logger','var') || tc.logging_logger(logger));
            assert(~exist('logger','var') || logger.active);
               
            if ~exist('logger','var')
                hnd_zero = logging.handlers.zero(logging.level.All);
                logger = logging.logger({hnd_zero});
            end
            
            logger.message('Opening images file "%s".',images_path);
            
            [images_fid,images_msg] = fopen(images_path,'rb');
            
            if images_fid == -1
                throw(MException('master:NoLoad',...
                         sprintf('Could not load images in "%s": %s!',images_path,images_msg)))
            end
            
            logger.message('Opening labels file "%s".',labels_path);
            
            [labels_fid,labels_msg] = fopen(labels_path,'rb');
            
            if labels_fid == -1
                fclose(images_fid);
                throw(MException('master:NoLoad',...
                         sprintf('Could not load labels in "%s": %s!',labels_path,labels_msg)))
            end
            
            try
                logger.message('Reading images file magic number.');

                images_magic = dataset.high2low(fread(images_fid,4,'uint8=>uint32'));
                
                if images_magic ~= 2051
                    throw(MException('master:NoLoad',...
                             sprintf('Images file "%s" not in MNIST format!',images_path)));
                end
                
                logger.message('Reading labels file magic number.');
                
                labels_magic = dataset.high2low(fread(labels_fid,4,'uint8=>uint32'));
                
                if labels_magic ~= 2049
                    throw(MException('master:NoLoad',...
                             sprintf('Labels file "%s" not in MNIST format!',labels_path)));
                end
                
                logger.beg_node('Reading images and labels count (should be equal)');
                
                images_count = dataset.high2low(fread(images_fid,4,'uint8=>uint32'));
                labels_count = dataset.high2low(fread(labels_fid,4,'uint8=>uint32'));
                
                if images_count ~= labels_count
                    throw(MException('master:NoLoad',...
                             sprintf('Different number of labels in "%s" for images in "%s"!',labels_path,images_path)));
                end
                
                logger.message('Images count: %d',images_count);
                logger.message('Labels count: %d',labels_count);
                
                logger.end_node();
                
                logger.beg_node('Reading images row and col count');
                
                row_count = dataset.high2low(fread(images_fid,4,'uint8=>uint32'));
                col_count = dataset.high2low(fread(images_fid,4,'uint8=>uint32'));
                
                logger.message('Row count: %d',row_count);
                logger.message('Col count: %d',col_count);
                
                logger.end_node();
                
                log_batch_size = ceil(images_count / 10);
                images = zeros(row_count,col_count,1,images_count);
                
                logger.beg_node('Starting reading of images');
                
                for ii = 1:images_count
                    if mod(ii - 1,log_batch_size) == 0
                        logger.message('Images %d to %d',ii,min(ii + log_batch_size - 1,images_count));
                    end
                    
                    images(:,:,1,ii) = fread(images_fid,[row_count col_count],'uint8=>double')' ./ 255;
                end
                
                logger.end_node();
                
                logger.message('Starting reading of labels');
                
                labels = fread(labels_fid,[images_count 1],'uint8');
                
                fclose(images_fid);
                fclose(labels_fid);
            catch exp
                fclose(images_fid);
                fclose(labels_fid);
                throw(MException('master:NoLoad',exp.message));
            end
            
            logger.message('Building dataset and labels information.');
            
            sample = images;
            class_info = classification_info({'d0' 'd1' 'd2' 'd3' 'd4' 'd5' 'd6' 'd7' 'd8' 'd9'},labels + 1);
        end
        
        function [new_sample] = rebuild_image(sample,layers_count,row_count,col_count)
            assert(tc.dataset_record(sample));
            assert(tc.scalar(layers_count));
            assert(tc.natural(layers_count));
            assert(layers_count >= 1);
            assert(tc.scalar(row_count));
            assert(tc.natural(row_count));
            assert(row_count >= 1);
            assert(tc.scalar(col_count));
            assert(tc.natural(col_count));
            assert(col_count >= 1);
            assert(size(sample,1) == (layers_count * row_count * col_count));
            
            N = dataset.count(sample);
            new_sample = reshape(sample,row_count,col_count,layers_count,N);
        end

        function [new_sample] = flatten_image(sample)
            assert(tc.dataset_image(sample));
            
            N = dataset.count(sample);
            [d,~,~,~] = dataset.geometry(sample);
            new_sample = reshape(sample,d,N);
        end

        function [new_sample] = subsample(sample,index)
            assert(tc.dataset(sample));
            assert(tc.vector(index));
            
            if tc.dataset_record(sample)
                assert((tc.logical(index) && tc.match_dims(sample,index,2,2)) || ...
                       (tc.natural(index) && tc.check(index >= 1 & index <= size(sample,2))));
                   
                new_sample = sample(:,index);
            elseif tc.dataset_image(sample)
                assert((tc.logical(index) && tc.match_dims(sample,index,4)) || ...
                       (tc.natural(index) && tc.check(index >= 1 & index <= size(sample,4))));
                   
                new_sample = sample(:,:,:,index);
            else
                assert(false);
            end
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
            fprintf('Testing "dataset".\n');
            
            fprintf('  Function "count".\n');
            
            s_1 = randi(2,4,10);
            s_2 = randi(2,8,8,1,10);
            
            assert(dataset.count(s_1) == 10);
            assert(dataset.count(s_2) == 10);
            
            clearvars -except display;
            
            fprintf('  Function "geometry".\n');
            
            fprintf('    Geometry of records.\n');
            
            s = randi(2,4,10);
            
            d = dataset.geometry(s);
            
            assert(d == 4);
            
            clearvars -except display;
            
            fprintf('    Geometry of images.\n');
            
            s = randi(2,8,4,3,100);
            
            g = dataset.geometry(s);
            
            assert(tc.same(g,[8*4*3 8 4 3]));
            
            [d,d_r,d_c,d_l] = dataset.geometry(s);
            
            assert(d == 8*4*3);
            assert(d_r == 8);
            assert(d_c == 4);
            assert(d_l == 3);
            
            clearvars -except display;
            
            fprintf('  Function "geom_compatible".\n');
            
            s_1 = rand(2,10);
            s_2 = rand(2,10);
            s_3 = rand(4,10);
            s_4 = rand(1,10);
            s_5 = rand(8,8,1,10);
            s_6 = rand(10,10,1,20);
            
            assert(dataset.geom_compatible(dataset.geometry(s_1),dataset.geometry(s_2)) == true);
            assert(dataset.geom_compatible(dataset.geometry(s_1),dataset.geometry(s_3)) == false);
            assert(dataset.geom_compatible(dataset.geometry(s_1),dataset.geometry(s_4)) == false);
            assert(dataset.geom_compatible(dataset.geometry(s_1),dataset.geometry(s_5)) == false);
            assert(dataset.geom_compatible(dataset.geometry(s_1),dataset.geometry(s_6)) == false);
            
            s_7 = rand(8,8,1,10);
            s_8 = rand(8,8,1,10);
            s_9 = rand(8,8,3,10);
            s_10 = rand(9,9,1,10);
            s_11 = rand(8,9,1,10);
            s_12 = rand(8,2);
            s_13 = rand(9,3);
            
            assert(dataset.geom_compatible(dataset.geometry(s_7),dataset.geometry(s_8)) == true);
            assert(dataset.geom_compatible(dataset.geometry(s_7),dataset.geometry(s_9)) == false);
            assert(dataset.geom_compatible(dataset.geometry(s_7),dataset.geometry(s_10)) == false);
            assert(dataset.geom_compatible(dataset.geometry(s_7),dataset.geometry(s_11)) == false);
            assert(dataset.geom_compatible(dataset.geometry(s_7),dataset.geometry(s_12)) == false);
            assert(dataset.geom_compatible(dataset.geometry(s_7),dataset.geometry(s_13)) == false);
            
            clearvars -except display;
            
            fprintf('  Function "load_record_csvfile".\n');
            
            fprintf('    With Wine data and "," delimiter (default).\n');
            
            [s,ci] = dataset.load_record_csvfile('../test/wine/wine.csv','%f%f%f%f%f%f%f%f%f%f%f%f%f');
            
            assert(tc.dataset_record(s));
            assert(tc.same(size(s),[13 178]));
            assert(length(ci.labels) == 3);
            assert(strcmp(ci.labels{1},'1'));
            assert(strcmp(ci.labels{2},'2'));
            assert(strcmp(ci.labels{3},'3'));
            assert(ci.labels_count == 3);
            assert(tc.same(ci.labels_idx,[1*ones(1,59) 2*ones(1,71) 3*ones(1,48)]));
            
            clearvars -except display;
            
            fprintf('    With iris data and "," delimiter.\n');
            
            [s,ci] = dataset.load_record_csvfile('../test/iris/iris.csv','%f%f%f%f',',');
            
            assert(tc.dataset_record(s));
            assert(tc.same(size(s),[4 150]));
            assert(length(ci.labels) == 3);
            assert(strcmp(ci.labels{1},'Iris-setosa'));
            assert(strcmp(ci.labels{2},'Iris-versicolor'));
            assert(strcmp(ci.labels{3},'Iris-virginica'));
            assert(ci.labels_count == 3);
            assert(tc.same(ci.labels_idx,[1*ones(1,50) 2*ones(1,50) 3*ones(1,50)]));
            
            clearvars -except display;
            
            fprintf('    With iris data and "," delimiter and valid logger.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = dataset.load_record_csvfile('../test/iris/iris.csv','%f%f%f%f',',',log);
            
            assert(tc.dataset_record(s));
            assert(tc.same(size(s),[4 150]));
            assert(length(ci.labels) == 3);
            assert(strcmp(ci.labels{1},'Iris-setosa'));
            assert(strcmp(ci.labels{2},'Iris-versicolor'));
            assert(strcmp(ci.labels{3},'Iris-virginica'));
            assert(ci.labels_count == 3);
            assert(tc.same(ci.labels_idx,[1*ones(1,50) 2*ones(1,50) 3*ones(1,50)]));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With invalid external inputs.\n');
            
            try
                dataset.load_record_csvfile('../test/wine/wine_aaa.csv','%d','%f%f%f%f%f%f%f%f%f%f%f%f%f');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not load csv file "../test/wine/wine_aaa.csv": No such file or directory!')
                    fprintf('      Passes "No such file or directory!" test.\n');
                else
                    assert(false);
                end
            end
            
            try
                chmod_code = system('chmod a-r ../test/wine/wine.csv');
                
                assert(chmod_code == 0);
                
                dataset.load_record_csvfile('../test/wine/wine.csv','%d','%f%f%f%f%f%f%f%f%f%f%f%f%f');
                
                chmod2_code = system('chmod a+r ../test/wine/wine.csv');
                
                assert(chmod2_code == 0);
                assert(false);
            catch exp
                chmod2_code = system('chmod a+r ../test/wine/wine.csv');
                
                assert(chmod2_code == 0);
                
                if strcmp(exp.message,'Could not load csv file "../test/wine/wine.csv": Permission denied!')
                    fprintf('      Passes "Permission denied!" test.\n');
                else
                    assert(false);
                end
            end
            
            try
                dataset.load_record_csvfile('../test/wine/wine.csv','%s%s%f%f%f%f%f%f%f%f%f%f%f');
                assert(false);
            catch exp
                if strcmp(exp.message,'File "../test/wine/wine.csv" has an invalid format!')
                    fprintf('      Passes "Invalid format!" test.\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except display;
            
            fprintf('  Function "load_image_from_dir".\n');
            
            fprintf('    With mode "gray" (default) and file size (default).\n');
            
            s = dataset.load_image_from_dir('../test/scenes_small');
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[192 256 1 7]));
            
            clearvars -except display;
            
            fprintf('    With mode "gray" and file size (default).\n');
            
            s = dataset.load_image_from_dir('../test/scenes_small','gray');
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[192 256 1 7]));
            
            clearvars -except display;
            
            fprintf('    With mode "original" and file size (default).\n');

            s = dataset.load_image_from_dir('../test/scenes_small','original');
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[192 256 3 7]));
            
            clearvars -except display;
            
            fprintf('    With mode "gray" and file size.\n');
            
            s = dataset.load_image_from_dir('../test/scenes_small','gray',[-1 -1]);
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[192 256 1 7]));

            clearvars -except display;
            
            fprintf('    With mode "original" and file size.\n');
            
            s = dataset.load_image_from_dir('../test/scenes_small','original',[-1 -1]);
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[192 256 3 7]));
            
            clearvars -except display;
            
            fprintf('    With mode "gray" and forced size.\n');
            
            s = dataset.load_image_from_dir('../test/scenes_small','gray',[96 128]);
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[96 128 1 7]));
            
            clearvars -except display;
            
            fprintf('    With mode "original" and forced size.\n');
            
            s = dataset.load_image_from_dir('../test/scenes_small','original',[96 128]);
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[96 128 3 7]));
            
            clearvars -except display;
            
            fprintf('    With mode "gray" and file size and a valid logger.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            s = dataset.load_image_from_dir('../test/scenes_small','gray',[-1 -1],log);
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[192 256 1 7]));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With mode "original" and file size and a valid logger.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            s = dataset.load_image_from_dir('../test/scenes_small','original',[-1 -1],log);
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[192 256 3 7]));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With mode "gray" and forced size and a valid logger.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            s = dataset.load_image_from_dir('../test/scenes_small','gray',[96 128],log);
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[96 128 1 7]));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With mode "original" and forced size and a valid logger.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            s = dataset.load_image_from_dir('../test/scenes_small','original',[96 128],log);
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[96 128 3 7]));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With heterogenous directory.\n');
            
            s = dataset.load_image_from_dir('../test/scenes_small/heterogeneous_dir');
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[192 256 1 2]));
            
            clearvars -except display;
            
            fprintf('    With invalid external inputs.\n');
            
            try
                dataset.load_image_from_dir('../test/scenes_small_aaa');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('      Passes "No such file or directory!"\n');
                else
                    assert(false);
                end
            end
            
            try
                chmod_code = system('chmod a-r ../test/scenes_small');
                
                assert(chmod_code == 0);
                
                dataset.load_image_from_dir('../test/scenes_small');
                
                chmod_code = system('chmod a+r ../test/scenes_small');
                
                assert(chmod_code == 0);
                assert(false);
            catch exp
                chmod_code = system('chmod a+r ../test/scenes_small');
                
                assert(chmod_code == 0);
                
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('      Passes "Permission denied!"\n');
                else
                    assert(false);
                end
            end
            
            try
                dataset.load_image_from_dir('../test/scenes_small/empty_dir');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('      Passes "Empty directory!"\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except display;
            
            fprintf('  Function "load_image_mnist".\n');
            
            fprintf('    With MNIST test data.\n');
            
            [s,ci] = dataset.load_image_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte');
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[28 28 1 10000]));
            assert(length(ci.labels) == 10);
            assert(strcmp(ci.labels{1},'d0'));
            assert(strcmp(ci.labels{2},'d1'));
            assert(strcmp(ci.labels{3},'d2'));
            assert(strcmp(ci.labels{4},'d3'));
            assert(strcmp(ci.labels{5},'d4'));
            assert(strcmp(ci.labels{6},'d5'));
            assert(strcmp(ci.labels{7},'d6'));
            assert(strcmp(ci.labels{8},'d7'));
            assert(strcmp(ci.labels{9},'d8'));
            assert(strcmp(ci.labels{10},'d9'));
            assert(ci.labels_count == 10);
            assert(tc.vector(ci.labels_idx));
            assert(tc.match_dims(s,ci.labels_idx,4));
            assert(tc.labels_idx(ci.labels_idx,ci.labels));
            
            clearvars -except display;
            
            fprintf('    With MNIST test data and a valid logger.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            
            [s,ci] = dataset.load_image_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte',log);
            
            assert(tc.dataset_image(s));
            assert(tc.unitreal(s));
            assert(tc.same(size(s),[28 28 1 10000]));
            assert(length(ci.labels) == 10);
            assert(strcmp(ci.labels{1},'d0'));
            assert(strcmp(ci.labels{2},'d1'));
            assert(strcmp(ci.labels{3},'d2'));
            assert(strcmp(ci.labels{4},'d3'));
            assert(strcmp(ci.labels{5},'d4'));
            assert(strcmp(ci.labels{6},'d5'));
            assert(strcmp(ci.labels{7},'d6'));
            assert(strcmp(ci.labels{8},'d7'));
            assert(strcmp(ci.labels{9},'d8'));
            assert(strcmp(ci.labels{10},'d9'));
            assert(ci.labels_count == 10);
            assert(tc.vector(ci.labels_idx));
            assert(tc.match_dims(s,ci.labels_idx,4));
            assert(tc.labels_idx(ci.labels_idx,ci.labels));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With invalid external inputs.\n');
            
            try
                dataset.load_image_mnist('../test/mnist/t10k-images-idx3-ubyte_aaa','../test/mnist/t10k-labels-idx1-ubyte');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not load images in "../test/mnist/t10k-images-idx3-ubyte_aaa": No such file or directory!')
                    fprintf('      Passes "No such file or directory!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                dataset.load_image_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte_aaa');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not load labels in "../test/mnist/t10k-labels-idx1-ubyte_aaa": No such file or directory!')
                    fprintf('      Passes "No such file or directory!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                chmod_code = system('chmod a-r ../test/mnist/t10k-images-idx3-ubyte');
                
                assert(chmod_code == 0);
                
                dataset.load_image_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte');
                
                chmod2_code = system('chmod a+r ../test/mnist/t10k-images-idx3-ubyte');
                
                assert(chmod2_code == 0);
                assert(false);
            catch exp
                chmod2_code = system('chmod a+r ../test/mnist/t10k-images-idx3-ubyte');
                
                assert(chmod2_code == 0);
                
                if strcmp(exp.message,'Could not load images in "../test/mnist/t10k-images-idx3-ubyte": Permission denied!')
                    fprintf('      Passes "Permission denied!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                chmod_code = system('chmod a-r ../test/mnist/t10k-labels-idx1-ubyte');
                
                assert(chmod_code == 0);
                
                dataset.load_image_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte');
                
                chmod2_code = system('chmod a+r ../test/mnist/t10k-labels-idx1-ubyte');
                
                assert(chmod2_code == 0);                
                assert(false);
            catch exp
                chmod2_code = system('chmod a+r ../test/mnist/t10k-labels-idx1-ubyte');
                
                assert(chmod2_code == 0);                
                
                if strcmp(exp.message,'Could not load labels in "../test/mnist/t10k-labels-idx1-ubyte": Permission denied!')
                    fprintf('      Passes "Permission denied!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                dataset.load_image_mnist('../test/scenes_small/scenes_small1.jpg','../test/mnist/t10k-labels-idx1-ubyte');
            catch exp
                if strcmp(exp.message,'Images file "../test/scenes_small/scenes_small1.jpg" not in MNIST format!')
                    fprintf('      Passes "Not in MNIST format!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                dataset.load_image_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/scenes_small/scenes_small1.jpg');
            catch exp
                if strcmp(exp.message,'Labels file "../test/scenes_small/scenes_small1.jpg" not in MNIST format!')
                    fprintf('      Passes "Not in MNIST format!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                dataset.load_image_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte');
            catch exp
                if strcmp(exp.message,'Different number of labels in "../test/mnist/t10k-labels-idx1-ubyte" for images in "../test/mnist/t10k-images-idx3-ubyte"!')
                    fprintf('      Passes "Different number of labels!".\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except display;
            
            fprintf('  Function "rebuild_image".\n');
            
            fprintf('    Single layer.\n');
            
            A = rand(100,20);
            A_i = zeros(10,10,1,20);
            for ii = 1:20
                A_i(:,:,1,ii) = reshape(A(:,ii),[10 10]);
            end
            
            s = dataset.rebuild_image(A,1,10,10);
            
            assert(tc.dataset_image(s));
            assert(tc.same(size(s),[10 10 1 20]));
            assert(tc.unitreal(s));
            assert(tc.same(s,A_i));
            
            clearvars -except display;
            
            fprintf('    Three layers.\n');
            
            A = rand(300,20);
            A_i = zeros(10,10,3,20);
            for ii = 1:20
                A_i(:,:,:,ii) = reshape(A(:,ii),[10 10 3]);
            end
            
            s = dataset.rebuild_image(A,3,10,10);
            
            assert(tc.dataset_image(s));
            assert(tc.same(size(s),[10 10 3 20]));
            assert(tc.unitreal(s));
            assert(tc.same(s,A_i));

            clearvars -except display;
            
            fprintf('  Function "flatten_image".\n');
            
            fprintf('    Single layer.\n');
            
            A = rand(10,10,1,20);
            A_i = zeros(100,20);
            for ii = 1:20
                A_i(:,ii) = reshape(A(:,:,:,ii),100,1);
            end
            
            s = dataset.flatten_image(A);
            
            assert(tc.dataset_record(s));
            assert(tc.same(size(s),[100 20]));
            assert(tc.unitreal(s));
            assert(tc.same(s,A_i));
            
            clearvars -except display;
            
            fprintf('    Three layers.\n');
            
            A = rand(10,10,3,20);
            A_i = zeros(300,20);
            for ii = 1:20
                A_i(:,ii) = reshape(A(:,:,:,ii),300,1);
            end
            
            s = dataset.flatten_image(A);
            
            assert(tc.dataset_record(s));
            assert(tc.same(size(s),[300 20]));
            assert(tc.unitreal(s));
            assert(tc.same(s,A_i));
            
            clearvars -except display;
            
            fprintf('  Function "subsample".\n');
            
            fprintf('    With boolean indices on records.\n');
            
            s = rand(10,100);
            idx = logical(randi(2,1,100) - 1);
            
            s_1 = dataset.subsample(s,idx);
            
            assert(tc.dataset_record(s_1));
            assert(tc.same(size(s_1),[10 sum(idx)]));
            assert(tc.unitreal(s_1));
            assert(tc.same(s_1,s(:,idx)));
            
            clearvars -except display;
            
            fprintf('    With boolean indices on images.\n');
            
            s = rand(8,8,3,100);
            idx = logical(randi(2,1,100) - 1);
            
            s_1 = dataset.subsample(s,idx);
            
            assert(tc.dataset_image(s_1));
            assert(tc.same(size(s_1),[8 8 3 sum(idx)]));
            assert(tc.unitreal(s_1));
            assert(tc.same(s_1,s(:,:,:,idx)));
            
            clearvars -except display;
            
            fprintf('    With integer indices on records.\n');
            
            s = rand(10,100);
            idx = randi(100,20,1);
            
            s_1 = dataset.subsample(s,idx);
            
            assert(tc.dataset_record(s_1));
            assert(tc.same(size(s_1),[10 20]));
            assert(tc.unitreal(s_1));
            assert(tc.same(s_1,s(:,idx)));
            
            clearvars -except display;
            
            fprintf('    With integer indices on images.\n');
            
            s = rand(8,8,3,100);
            idx = randi(100,10,1);
            
            s_1 = dataset.subsample(s,idx);
            
            assert(tc.dataset_image(s_1));
            assert(tc.same(size(s_1),[8 8 3 10]));
            assert(tc.unitreal(s_1));
            assert(tc.same(s_1,s(:,:,:,idx)));
            
            clearvars -except display;
        end
    end
end
