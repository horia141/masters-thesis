classdef gray_images_set < samples_set
    properties (GetAccess=public,SetAccess=immutable)
        images;
    end
    
    properties (GetAccess=public,SetAccess=private,Dependent=true)
        row_count;
        col_count;
    end
    
    methods
        function [row_count] = get.row_count(obj)
            row_count = size(obj.images,1);
        end
        
        function [col_count] = get.col_count(obj)
            col_count = size(obj.images,2);
        end
    end
    
    methods (Access=public)
        function [obj] = gray_images_set(classes,images,labels_idx)
            assert(tc.vector(classes) && tc.cell(classes));
            assert(tc.tensor(images,3) && tc.unitreal(images));
            assert(tc.vector(labels_idx) && tc.match_dims(images,labels_idx,3) && ...
                   tc.labels_idx(labels_idx,classes));
            
            obj = obj@samples_set(classes,gray_images_set.to_samples(images),labels_idx);
            obj.images = images;
        end
        
        function [new_gray_images_set] = subsamples(obj,index)
            assert(tc.vector(index) && ...
                   ((tc.logical(index) && tc.match_dims(obj.images,index,3)) || ...
                    (tc.natural(index) && tc.check(index > 0 & index <= obj.samples_count))));
                
            new_gray_images_set = gray_images_set(obj.classes,obj.images(:,:,index),obj.labels_idx(index));
        end
    end
    
    methods (Static,Access=public)
        function [new_gray_images_set] = from_data(images,labels)
            assert(tc.tensor(images,3) && tc.unitreal(images));
            assert(tc.vector(labels) && tc.match_dims(images,labels,3) && tc.labels(labels));
            
            [labels_idx_t,classes_t] = grp2idx(labels);
            new_gray_images_set = gray_images_set(classes_t,images,labels_idx_t);
        end
        
        function [new_gray_images_set] = load_from_dir(images_dir_path,force_size)
            assert(tc.string(images_dir_path));
            assert(~exist('force_size','var') || ...
                   (tc.vector(force_size) && (length(force_size) == 2) && ...
                    tc.natural(force_size) && tc.check(force_size > 1)));
               
            paths = dir(images_dir_path);
            images_t = [];
            current_image = 1;
            
            for i = 1:length(paths)
                try
                    image = imread(fullfile(images_dir_path,paths(i).name));
                    image = double(rgb2gray(image)) ./ 255;
                    
                    if exist('force_size','var')
                        image = imresize(image,force_size);
                        
                        % Correct domain overflows caused by resizing.
                        image = min(image,1);
                        image = max(image,0);
                    end
                    
                    if (current_image > 1) && ...
                        (~all(size(image) == size(images_t(:,:,1))))
                        throw(MException('master:Images:load_from_dir:NoLoad',...
                                         'Images are of different sizes!'));
                    end

                    images_t(:,:,current_image) = image;
                    current_image = current_image + 1;
                catch exp
                    if isempty(regexp(exp.identifier,'MATLAB:imread:.*','ONCE'))
                        throw(MException('master:Images:load_from_dir:NoLoad',exp.message));
                    end
                end
            end
            
            if isempty(images_t)
                throw(MException('master:Images:load_from_dir:NoLoad',...
                                 'Could not find any acceptable images in the directory.'));
            end
            
            new_gray_images_set = gray_images_set({'0','1'},images_t,ones(current_image - 1,1));
        end
        
        function [new_gray_images_set] = load_mnist(images_path,labels_path)
            assert(tc.string(images_path));
            assert(tc.string(labels_path));
            
            [images_fid,images_msg] = fopen(images_path,'rb');
            
            if images_fid == -1
                throw(MException('master:Images:load_mnist:NoLoad',...
                    sprintf('Could not load images in "%s": %s!',images_path,images_msg)))
            end
            
            [labels_fid,labels_msg] = fopen(labels_path,'rb');
            
            if labels_fid == -1
                fclose(images_fid);
                throw(MException('master:Images:load_mnist:NoLoad',...
                    sprintf('Could not load labels in "%s": %s!',labels_path,labels_msg)))
            end
            
            try
                images_magic = gray_images_set.high2low(fread(images_fid,4,'uint8=>uint32'));
                
                if images_magic ~= 2051
                    throw(MException('master:Images:load_mnist:NoLoad',...
                        sprintf('Images file "%s" not in MNIST format!',images_path)));
                end
                
                labels_magic = gray_images_set.high2low(fread(labels_fid,4,'uint8=>uint32'));
                
                if labels_magic ~= 2049
                    throw(MException('master:Images:load_mnist:NoLoad',...
                        sprintf('Labels file "%s" not in MNIST format!',labels_path)));
                end
                
                images_count = gray_images_set.high2low(fread(images_fid,4,'uint8=>uint32'));
                labels_count = gray_images_set.high2low(fread(labels_fid,4,'uint8=>uint32'));
                
                if images_count ~= labels_count
                    throw(MException('master:Images:load_mnist:NoLoad',...
                        sprintf('Different number of labels in "%s" for images in "%s"!',labels_path,images_path)));
                end
                
                row_count_t = gray_images_set.high2low(fread(images_fid,4,'uint8=>uint32'));
                col_count_t = gray_images_set.high2low(fread(images_fid,4,'uint8=>uint32'));
                
                images_t = zeros(row_count_t,col_count_t,images_count);
                
                for i = 1:images_count
                    images_t(:,:,i) = fread(images_fid,[row_count_t col_count_t],'uint8=>double')' ./ 255;
                end
                
                labels_t = fread(labels_fid,[images_count 1],'uint8');
                
                fclose(images_fid);
                fclose(labels_fid);
            catch exp
                fclose(images_fid);
                fclose(labels_fid);
                throw(MException('master:Images:load_mnist:NoLoad',exp.message));
            end

            new_gray_images_set = gray_images_set.from_data(images_t,labels_t);
        end
    end
    
    methods (Static,Access=private)
        function [out] = high2low(bytes)
            out = bitshift(bytes(4),0) + bitshift(bytes(3),8) + ...
                  bitshift(bytes(2),16) + bitshift(bytes(1),24);
        end
        
        function [samples] = to_samples(images)
            features_count = size(images,1) * size(images,2);
            samples = zeros(size(images,3),features_count);
            
            for i = 1:size(images,3)
                samples(i,:) = reshape(images(:,:,i),[1 features_count]);
            end
        end
    end
    
    methods (Static,Access=public)        
        function test
            fprintf('Testing "gray_images_set".\n');
            
            % Try a normal run first. Just build an object using the
            % constructor and call the two possible methods ("partition"
            % and "subsamples"). In all cases, see if we obtain correct
            % results, that is the internal representation matches what we
            % would expect, given our data.
            
            fprintf('  Testing proper construction, "partition" and "subsamples".\n');
            
            A = rand(10,10,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,100);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,i),[100 1]);
            end
                        
            s1 = gray_images_set({'1' '2' '3'},A,c);
            
            assert(length(s1.classes) == 3);
            assert(strcmp(s1.classes(1),'1'));
            assert(strcmp(s1.classes(2),'2'));
            assert(strcmp(s1.classes(3),'3'));
            assert(s1.classes_count == 3);
            assert(all(size(s1.samples) == [12 100]));
            assert(all(all(s1.samples == A_s)));
            assert(length(s1.labels_idx) == 12);
            assert(all(s1.labels_idx == c'));
            assert(s1.samples_count == 12);
            assert(s1.features_count == 100);
            for i = 1:12
                assert(all(all(s1.images(:,:,i) == A(:,:,i))));
            end
            assert(s1.row_count == 10);
            assert(s1.col_count == 10);
            assert(all(all(s1.samples == gray_images_set.to_samples(s1.images))));
            
            [tr_f,ts_f] = s1.partition('kfold',2);
            
            s1_f11 = s1.subsamples(tr_f(:,1));
            
            assert(length(s1_f11.classes) == 3);
            assert(strcmp(s1_f11.classes(1),'1'));
            assert(strcmp(s1_f11.classes(2),'2'));
            assert(strcmp(s1_f11.classes(3),'3'));
            assert(s1_f11.classes_count == 3);
            assert(all(size(s1_f11.samples) == [6 100]));
            assert(all(all(s1_f11.samples == A_s(tr_f(:,1),:))));
            assert(length(s1_f11.labels_idx) == 6);
            assert(all(s1_f11.labels_idx == c(tr_f(:,1))'));
            assert(s1_f11.samples_count == 6);
            assert(s1_f11.features_count == 100);
            j = 1;
            for i = 1:length(tr_f(:,1))
                if tr_f(i,1)
                    assert(all(all(s1_f11.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s1_f11.row_count == 10);
            assert(s1_f11.col_count == 10);
            assert(all(all(s1_f11.samples == gray_images_set.to_samples(s1_f11.images))));
            
            clear s1_f11
            
            s1_f12 = s1.subsamples(ts_f(:,1));
            
            assert(length(s1_f12.classes) == 3);
            assert(strcmp(s1_f12.classes(1),'1'));
            assert(strcmp(s1_f12.classes(2),'2'));
            assert(strcmp(s1_f12.classes(3),'3'));
            assert(s1_f12.classes_count == 3);
            assert(all(size(s1_f12.samples) == [6 100]));
            assert(all(all(s1_f12.samples == A_s(ts_f(:,1),:))));
            assert(length(s1_f12.labels_idx) == 6);
            assert(all(s1_f12.labels_idx == c(ts_f(:,1))'));
            assert(s1_f12.samples_count == 6);
            assert(s1_f12.features_count == 100);
            j = 1;
            for i = 1:length(ts_f(:,1))
                if ts_f(i,1)
                    assert(all(all(s1_f12.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s1_f12.row_count == 10);
            assert(s1_f12.col_count == 10);
            assert(all(all(s1_f12.samples == gray_images_set.to_samples(s1_f12.images))));
            
            clear s1_f12
            
            s1_f21 = s1.subsamples(tr_f(:,2));

            assert(length(s1_f21.classes) == 3);
            assert(strcmp(s1_f21.classes(1),'1'));
            assert(strcmp(s1_f21.classes(2),'2'));
            assert(strcmp(s1_f21.classes(3),'3'));
            assert(s1_f21.classes_count == 3);
            assert(all(size(s1_f21.samples) == [6 100]));
            assert(all(all(s1_f21.samples == A_s(tr_f(:,2),:))));
            assert(length(s1_f21.labels_idx) == 6);
            assert(all(s1_f21.labels_idx == c(tr_f(:,2))'));
            assert(s1_f21.samples_count == 6);
            assert(s1_f21.features_count == 100);
            j = 1;
            for i = 1:length(tr_f(:,2))
                if tr_f(i,2)
                    assert(all(all(s1_f21.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s1_f21.row_count == 10);
            assert(s1_f21.col_count == 10);
            assert(all(all(s1_f21.samples == gray_images_set.to_samples(s1_f21.images))));
            
            clear s1_f21
            
            s1_f22 = s1.subsamples(ts_f(:,2));
            
            assert(length(s1_f22.classes) == 3);
            assert(strcmp(s1_f22.classes(1),'1'));
            assert(strcmp(s1_f22.classes(2),'2'));
            assert(strcmp(s1_f22.classes(3),'3'));
            assert(s1_f22.classes_count == 3);
            assert(all(size(s1_f22.samples) == [6 100]));
            assert(all(all(s1_f22.samples == A_s(ts_f(:,2),:))));
            assert(length(s1_f22.labels_idx) == 6);
            assert(all(s1_f22.labels_idx == c(ts_f(:,2))'));
            assert(s1_f22.samples_count == 6);
            assert(s1_f22.features_count == 100);
            j = 1;
            for i = 1:length(ts_f(:,2))
                if ts_f(i,2)
                    assert(all(all(s1_f22.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s1_f22.row_count == 10);
            assert(s1_f22.col_count == 10);
            assert(all(all(s1_f22.samples == gray_images_set.to_samples(s1_f22.images))));
            
            clear s1_f22
            clear tr_f
            clear ts_f
            
            [tr_h,ts_h] = s1.partition('holdout',0.33);
            
            s1_h1 = s1.subsamples(tr_h);
            
            assert(length(s1_h1.classes) == 3);
            assert(strcmp(s1_h1.classes(1),'1'));
            assert(strcmp(s1_h1.classes(2),'2'));
            assert(strcmp(s1_h1.classes(2),'2'));
            assert(s1_h1.classes_count == 3);
            assert(all(size(s1_h1.samples) == [9 100]));
            assert(all(all(s1_h1.samples == A_s(tr_h,:))));
            assert(length(s1_h1.labels_idx) == 9);
            assert(all(s1_h1.labels_idx == c(tr_h)'));
            assert(s1_h1.samples_count == 9);
            assert(s1_h1.features_count == 100);
            j = 1;
            for i = 1:length(tr_h)
                if tr_h(i)
                    assert(all(all(s1_h1.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s1_h1.row_count == 10);
            assert(s1_h1.col_count == 10);
            assert(all(all(s1_h1.samples == gray_images_set.to_samples(s1_h1.images))));
            
            clear s1_h1
            
            s1_h2 = s1.subsamples(ts_h);
            
            assert(length(s1_h2.classes) == 3);
            assert(strcmp(s1_h2.classes(1),'1'));
            assert(strcmp(s1_h2.classes(2),'2'));
            assert(strcmp(s1_h2.classes(2),'2'));
            assert(s1_h2.classes_count == 3);
            assert(all(size(s1_h2.samples) == [3 100]));
            assert(all(all(s1_h2.samples == A_s(ts_h,:))));
            assert(length(s1_h2.labels_idx) == 3);
            assert(all(s1_h2.labels_idx == c(ts_h)'));
            assert(s1_h2.samples_count == 3);
            assert(s1_h2.features_count == 100);
            j = 1;
            for i = 1:length(ts_h)
                if ts_h(i)
                    assert(all(all(s1_h2.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s1_h2.row_count == 10);
            assert(s1_h2.col_count == 10);
            assert(all(all(s1_h2.samples == gray_images_set.to_samples(s1_h2.images))));
            
            clear s1_h2
            clear tr_h
            clear ts_h
            
            s1_fi = s1.subsamples(1:2:12);
            
            assert(length(s1_fi.classes) == 3);
            assert(strcmp(s1_fi.classes(1),'1'));
            assert(strcmp(s1_fi.classes(2),'2'));
            assert(strcmp(s1_fi.classes(3),'3'));
            assert(s1_fi.classes_count == 3);
            assert(all(size(s1_fi.samples) == [6 100]));
            assert(all(all(s1_fi.samples == A_s(1:2:12,:))));
            assert(length(s1_fi.labels_idx) == 6);
            assert(all(s1_fi.labels_idx == c(1:2:12)'));
            assert(s1_fi.samples_count == 6);
            assert(s1_fi.features_count == 100);
            j = 1;
            for i = 1:2:12
                assert(all(all(s1_fi.images(:,:,j) == A(:,:,i))));
                j = j + 1;
            end
            assert(s1_fi.row_count == 10);
            assert(s1_fi.col_count == 10);
            assert(all(all(s1_fi.samples == gray_images_set.to_samples(s1_fi.images))));
            
            clear s1_fi
            
            s1_fo = s1.subsamples([1:12,1:12]);
            
            assert(length(s1_fo.classes) == 3);
            assert(strcmp(s1_fo.classes(1),'1'));
            assert(strcmp(s1_fo.classes(2),'2'));
            assert(strcmp(s1_fo.classes(3),'3'));
            assert(s1_fo.classes_count == 3);
            assert(all(size(s1_fo.samples) == [24 100]));
            assert(all(all(s1_fo.samples == [A_s;A_s])));
            assert(length(s1_fo.labels_idx) == 24);
            assert(all(s1_fo.labels_idx == [c c]'));
            assert(s1_fo.samples_count == 24);
            assert(s1_fo.features_count == 100);
            j = 1;
            for i = 0:23
                assert(all(all(s1_fo.images(:,:,j) == A(:,:,mod(i,12) + 1))));
                j = j + 1;
            end
            assert(s1_fo.row_count == 10);
            assert(s1_fo.col_count == 10);
            assert(all(all(s1_fo.samples == gray_images_set.to_samples(s1_fo.images))));
            
            clear s1_fo
            
            clear s1
            
            % Try building from pre-existing data using the "from_data"
            % static method.
            
            fprintf('  Testing function "from_data".\n');
            
            A = rand(10,10,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,100);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,i),[100 1]);
            end
            
            s2 = gray_images_set.from_data(A,c);
            
            assert(length(s2.classes) == 3);
            assert(strcmp(s2.classes(1),'1'));
            assert(strcmp(s2.classes(2),'2'));
            assert(strcmp(s2.classes(3),'3'));
            assert(s2.classes_count == 3);
            assert(all(size(s2.samples) == [12 100]));
            assert(all(all(s2.samples == A_s)));
            assert(length(s2.labels_idx) == 12);
            assert(all(s2.labels_idx == c'));
            assert(s2.samples_count == 12);
            assert(s2.features_count == 100);
            for i = 1:12
                assert(all(all(s2.images(:,:,i) == A(:,:,i))));
            end
            assert(s2.row_count == 10);
            assert(s2.col_count == 10);
            assert(all(all(s2.samples == gray_images_set.to_samples(s2.images))));
            
            clear s2
            
            % Try loading files from a directory. The files are stored in
            % "$PROJECT_ROOT/data/test/". This directory should
            % exist in all distributions of this project.
            
            fprintf('  Testing "load_from_dir" with test data.\n');
            
            s3 = gray_images_set.load_from_dir('../data/test');
            
            assert(length(s3.classes) == 2);
            assert(strcmp(s3.classes(1),'0'));
            assert(strcmp(s3.classes(2),'1'));
            assert(s3.classes_count == 2);
            assert(all(size(s3.samples) == [7 192*256]));
            assert(tc.matrix(s3.samples) && tc.unitreal(s3.samples));
            assert(all(size(s3.labels_idx) == [7 1]));
            assert(all(s3.labels_idx == ones(7,1)));
            assert(s3.samples_count == 7);
            assert(s3.features_count == 192*256);
            assert(tc.tensor(s3.images,3) && tc.unitreal(s3.images));
            assert(s3.row_count == 192);
            assert(s3.col_count == 256);
            assert(all(all(s3.samples == gray_images_set.to_samples(s3.images))));
            
            clear s3
            
            % Try loading files from a directory, with forced size. The 
            % files are stored in "$PROJECT_ROOT/data/test/". This 
            % directory should exist in all distributions of this project.
            
            fprintf('  Testing "load_from_dir" with forced size on test data.\n');
            
            s4 = gray_images_set.load_from_dir('../data/test',[96 128]);
            
            assert(length(s4.classes) == 2);
            assert(strcmp(s4.classes(1),'0'));
            assert(strcmp(s4.classes(2),'1'));
            assert(s4.classes_count == 2);
            assert(all(size(s4.samples) == [7 96*128]));
            assert(tc.matrix(s4.samples) && tc.unitreal(s4.samples));
            assert(all(size(s4.labels_idx) == [7 1]));
            assert(all(s4.labels_idx == ones(7,1)));
            assert(s4.samples_count == 7);
            assert(s4.features_count == 96*128);
            assert(tc.tensor(s4.images,3) && tc.unitreal(s4.images));
            assert(s4.row_count == 96);
            assert(s4.col_count == 128);
            assert(all(all(s4.samples == gray_images_set.to_samples(s4.images))));
            
            clear s4
            
            % Try loading files from a directory which contains other types
            % of files besides images. The files are stored in
            % "$PROJECT_ROOT/data/test/heterogeneous_dir". This directory
            % should exits for all distributions of this project.
            
            fprintf('  Testing "load_from_dir" with heterogenous directory.\n');
            
            s5 = gray_images_set.load_from_dir('../data/test/heterogeneous_dir');
            
            assert(length(s5.classes) == 2);
            assert(strcmp(s5.classes(1),'0'));
            assert(strcmp(s5.classes(2),'1'));
            assert(s5.classes_count == 2);
            assert(all(size(s5.samples) == [2 192*256]));
            assert(tc.matrix(s5.samples) && tc.unitreal(s5.samples));
            assert(all(size(s5.labels_idx) == [2 1]));
            assert(all(s5.labels_idx == ones(2,1)));
            assert(s5.samples_count == 2);
            assert(s5.features_count == 192*256);
            assert(tc.tensor(s5.images,3) && tc.unitreal(s5.images));
            assert(s5.row_count == 192);
            assert(s5.col_count == 256);
            assert(all(all(s5.samples == gray_images_set.to_samples(s5.images))));
            
            clear s5
            
            % Try some invalid calls to "load_from_dir". These test the
            % failure modes of this function. We're interested in things
            % beyond the caller's control, like empty directories,
            % insufficient access rights and images of different sizes.
            
            fprintf('  Testing "load_from_dir" with invalid external inputs.\n');
            
            try
                s6 = gray_images_set.load_from_dir('../data/test_aaa');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('    Passes "No such file or directory!"\n');
                else
                    assert(false);
                end
            end
            
            try
                !chmod a-r ../data/test
                s7 = gray_images_set.load_from_dir('../data/test');
                !chmod a+r ../data/test
                assert(false);
            catch exp
                !chmod a+r ../data/test
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('    Passes "Permission denied!"\n');
                else
                    assert(false);
                end
            end
            
            try
                s8 = gray_images_set.load_from_dir('../data/test/empty_dir');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('    Passes "Empty directory!"\n');
                else
                    assert(false);
                end
            end
            
            % Try loading images saved in the MNIST format. The images are
            % stored in "$PROJECT_ROOT/data/mnist/train-images-idx3-ubyte",
            % while the labels are stored in
            % "$PROJECT_ROOT/data/mnist/train-labels-idx1-ubyte". These two
            % files should exist in all distributions of this project.
            
            fprintf('  Testing "load_mnist" with MNIST training data.\n');
            
            s9 = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte');
            
            assert(length(s9.classes) == 10);
            assert(strcmp(s9.classes(1),'0'));
            assert(strcmp(s9.classes(2),'1'));
            assert(strcmp(s9.classes(3),'2'));
            assert(strcmp(s9.classes(4),'3'));
            assert(strcmp(s9.classes(5),'4'));
            assert(strcmp(s9.classes(6),'5'));
            assert(strcmp(s9.classes(7),'6'));
            assert(strcmp(s9.classes(8),'7'));
            assert(strcmp(s9.classes(9),'8'));
            assert(strcmp(s9.classes(10),'9'));
            assert(s9.classes_count == 10);
            assert(all(size(s9.samples) == [60000 28*28]));
            assert(tc.matrix(s9.samples) && tc.unitreal(s9.samples));
            assert(all(size(s9.labels_idx) == [60000 1]));
            assert(tc.vector(s9.labels_idx) && tc.labels_idx(s9.labels_idx,s9.classes));
            assert(s9.samples_count == 60000);
            assert(s9.features_count == 28*28);
            assert(tc.tensor(s9.images,3) && tc.unitreal(s9.images));
            assert(s9.row_count == 28);
            assert(s9.col_count == 28);
            assert(all(all(s9.samples == gray_images_set.to_samples(s9.images))));
            
            % Try some invalid calls to "load_mnist". These test the
            % failure modes of this function. We're interested in things
            % beyond the caller's control, like improper access rights to
            % input files or badly formatted files. We will not test the
            % latter error path in detail though, as it is highly
            % improbable given our current setup. Maybe later, when other
            % MNIST like datasets are added, we should test what happens if
            % not enough images are present or any of a number of binary
            % format decoding errors.
            
            fprintf('  Testing "load_mnist" with invalid external inputs.\n');
            
            try
                s10 = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte_aaa','../data/mnist/train-labels-idx1-ubyte');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not load images in "../data/mnist/train-images-idx3-ubyte_aaa": No such file or directory!')
                    fprintf('    Passes "No such file or directory!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s11 = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte_aaa');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not load labels in "../data/mnist/train-labels-idx1-ubyte_aaa": No such file or directory!')
                    fprintf('    Passes "No such file or directory!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                !chmod a-r ../data/mnist/train-images-idx3-ubyte
                s12 = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte');
                !chmod a+r ../data/mnist/train-images-idx3-ubyte
                assert(false);
            catch exp
                !chmod a+r ../data/mnist/train-images-idx3-ubyte
                if strcmp(exp.message,'Could not load images in "../data/mnist/train-images-idx3-ubyte": Permission denied!')
                    fprintf('    Passes "Permission denied!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                !chmod a-r ../data/mnist/train-labels-idx1-ubyte
                s13 = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte');
                !chmod a+r ../data/mnist/train-labels-idx1-ubyte
                assert(false);
            catch exp
                !chmod a+r ../data/mnist/train-labels-idx1-ubyte
                if strcmp(exp.message,'Could not load labels in "../data/mnist/train-labels-idx1-ubyte": Permission denied!')
                    fprintf('    Passes "Permission denied!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s14 = gray_images_set.load_mnist('../data/test/scenes_small1.jpg','../data/mnist/train-labels-idx1-ubyte');
            catch exp
                if strcmp(exp.message,'Images file "../data/test/scenes_small1.jpg" not in MNIST format!')
                    fprintf('    Passes "Not in MNIST format!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s15 = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/test/scenes_small1.jpg');
            catch exp
                if strcmp(exp.message,'Labels file "../data/test/scenes_small1.jpg" not in MNIST format!')
                    fprintf('    Passes "Not in MNIST format!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s16 = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/t10k-labels-idx1-ubyte');
            catch exp
                if strcmp(exp.message,'Different number of labels in "../data/mnist/t10k-labels-idx1-ubyte" for images in "../data/mnist/train-images-idx3-ubyte"!')
                    fprintf('    Passes "Different number of labels!".\n');
                else
                    assert(false);
                end
            end
        end
    end
end
