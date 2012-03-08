classdef gray_images_set < samples_set
    properties (GetAccess=public,SetAccess=immutable)
        images;
        row_count;
        col_count;
    end
    
    methods (Access=public)
        function [obj] = gray_images_set(classes,images,labels_idx)
            assert(tc.vector(classes) && tc.cell(classes));
            assert(tc.tensor(images,3) && tc.unitreal(images));
            assert(tc.vector(labels_idx) && tc.match_dims(images,labels_idx,3) && ...
                   tc.labels_idx(labels_idx,classes));
            
            obj = obj@samples_set(classes,gray_images_set.to_samples(images),labels_idx);
            obj.images = images;
            obj.row_count = size(images,1);
            obj.col_count = size(images,2);
        end
        
        function [new_gray_images_set] = subsamples(obj,index)
            assert(tc.vector(index) && ...
                   ((tc.logical(index) && tc.match_dims(obj.images,index,3)) || ...
                    (tc.natural(index) && tc.check(index > 0 & index <= obj.samples_count))));
                
            new_gray_images_set = gray_images_set(obj.classes,obj.images(:,:,index),obj.labels_idx(index));
        end
    end
    
    methods (Static,Access=public)
        function [new_gray_images_set] = from_samples(samples,row_count,col_count,remap_type,remap_mode)
            assert(tc.samples_set(samples));
            assert(tc.scalar(row_count) && tc.natural(row_count) && (row_count > 0));
            assert(tc.scalar(col_count) && tc.natural(col_count) && (col_count > 0));
            assert(~exist('remap_type','var') || (tc.string(remap_type) && ...
                    (strcmp(remap_type,'none') || strcmp(remap_type,'clamp') || strcmp(remap_type,'remap'))));
            assert(~(exist('remap_type','var') && exist('remap_mode','var')) || ...
                    (strcmp(remap_type,'remap') && (tc.string(remap_mode) && (strcmp(remap_mode,'local') || strcmp(remap_mode,'global')))));
                
            if exist('remap_type','var')
                remap_type_t = remap_type;
            else
                remap_type_t = 'none';
            end
            
            if exist('remap_mode','var')
                remap_mode_t = remap_mode;
            else
                remap_mode_t = 'local';
            end
            
            new_gray_images_set_t1 = zeros(row_count,col_count,samples.samples_count);
            
            for i = 1:samples.samples_count
                new_gray_images_set_t1(:,:,i) = reshape(samples.samples(i,:),[row_count col_count]);
            end
            
            if strcmp(remap_type_t,'none')
                new_gray_images_set_t2 = new_gray_images_set_t1;
            elseif strcmp(remap_type_t,'clamp')
                new_gray_images_set_t2 = utils.clamp_images_to_unit(new_gray_images_set_t1);
            else
                new_gray_images_set_t2 = utils.remap_images_to_unit(new_gray_images_set_t1,remap_mode_t);
            end
            
            new_gray_images_set = gray_images_set(samples.classes,new_gray_images_set_t2,samples.labels_idx);
        end
        
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
                        
                        % Correct small domain overflows caused by resizing.
                        image = utils.remap_images_to_unit(image);
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
            
            new_gray_images_set = gray_images_set({'none'},images_t,ones(current_image - 1,1));
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
            
            fprintf('    Construction.\n');
            
            A = rand(10,10,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,100);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,i),[100 1]);
            end
                        
            s = gray_images_set({'1' '2' '3'},A,c);
            
            assert(length(s.classes) == 3);
            assert(strcmp(s.classes(1),'1'));
            assert(strcmp(s.classes(2),'2'));
            assert(strcmp(s.classes(3),'3'));
            assert(s.classes_count == 3);
            assert(all(size(s.samples) == [12 100]));
            assert(all(all(s.samples == A_s)));
            assert(length(s.labels_idx) == 12);
            assert(all(s.labels_idx == c'));
            assert(s.samples_count == 12);
            assert(s.features_count == 100);
            for i = 1:12
                assert(all(all(s.images(:,:,i) == A(:,:,i))));
            end
            assert(s.row_count == 10);
            assert(s.col_count == 10);
            assert(all(all(s.samples == gray_images_set.to_samples(s.images))));
            
            fprintf('    2-fold partition and call to "subsamples" with boolean indices.\n');
            
            [tr_f,ts_f] = s.partition('kfold',2);
            
            s_f11 = s.subsamples(tr_f(:,1));
            
            assert(length(s_f11.classes) == 3);
            assert(strcmp(s_f11.classes(1),'1'));
            assert(strcmp(s_f11.classes(2),'2'));
            assert(strcmp(s_f11.classes(3),'3'));
            assert(s_f11.classes_count == 3);
            assert(all(size(s_f11.samples) == [6 100]));
            assert(all(all(s_f11.samples == A_s(tr_f(:,1),:))));
            assert(length(s_f11.labels_idx) == 6);
            assert(all(s_f11.labels_idx == c(tr_f(:,1))'));
            assert(s_f11.samples_count == 6);
            assert(s_f11.features_count == 100);
            j = 1;
            for i = 1:length(tr_f(:,1))
                if tr_f(i,1)
                    assert(all(all(s_f11.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s_f11.row_count == 10);
            assert(s_f11.col_count == 10);
            assert(all(all(s_f11.samples == gray_images_set.to_samples(s_f11.images))));
            
            s_f12 = s.subsamples(ts_f(:,1));
            
            assert(length(s_f12.classes) == 3);
            assert(strcmp(s_f12.classes(1),'1'));
            assert(strcmp(s_f12.classes(2),'2'));
            assert(strcmp(s_f12.classes(3),'3'));
            assert(s_f12.classes_count == 3);
            assert(all(size(s_f12.samples) == [6 100]));
            assert(all(all(s_f12.samples == A_s(ts_f(:,1),:))));
            assert(length(s_f12.labels_idx) == 6);
            assert(all(s_f12.labels_idx == c(ts_f(:,1))'));
            assert(s_f12.samples_count == 6);
            assert(s_f12.features_count == 100);
            j = 1;
            for i = 1:length(ts_f(:,1))
                if ts_f(i,1)
                    assert(all(all(s_f12.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s_f12.row_count == 10);
            assert(s_f12.col_count == 10);
            assert(all(all(s_f12.samples == gray_images_set.to_samples(s_f12.images))));
            
            s_f21 = s.subsamples(tr_f(:,2));

            assert(length(s_f21.classes) == 3);
            assert(strcmp(s_f21.classes(1),'1'));
            assert(strcmp(s_f21.classes(2),'2'));
            assert(strcmp(s_f21.classes(3),'3'));
            assert(s_f21.classes_count == 3);
            assert(all(size(s_f21.samples) == [6 100]));
            assert(all(all(s_f21.samples == A_s(tr_f(:,2),:))));
            assert(length(s_f21.labels_idx) == 6);
            assert(all(s_f21.labels_idx == c(tr_f(:,2))'));
            assert(s_f21.samples_count == 6);
            assert(s_f21.features_count == 100);
            j = 1;
            for i = 1:length(tr_f(:,2))
                if tr_f(i,2)
                    assert(all(all(s_f21.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s_f21.row_count == 10);
            assert(s_f21.col_count == 10);
            assert(all(all(s_f21.samples == gray_images_set.to_samples(s_f21.images))));
            
            s_f22 = s.subsamples(ts_f(:,2));
            
            assert(length(s_f22.classes) == 3);
            assert(strcmp(s_f22.classes(1),'1'));
            assert(strcmp(s_f22.classes(2),'2'));
            assert(strcmp(s_f22.classes(3),'3'));
            assert(s_f22.classes_count == 3);
            assert(all(size(s_f22.samples) == [6 100]));
            assert(all(all(s_f22.samples == A_s(ts_f(:,2),:))));
            assert(length(s_f22.labels_idx) == 6);
            assert(all(s_f22.labels_idx == c(ts_f(:,2))'));
            assert(s_f22.samples_count == 6);
            assert(s_f22.features_count == 100);
            j = 1;
            for i = 1:length(ts_f(:,2))
                if ts_f(i,2)
                    assert(all(all(s_f22.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s_f22.row_count == 10);
            assert(s_f22.col_count == 10);
            assert(all(all(s_f22.samples == gray_images_set.to_samples(s_f22.images))));
            
            fprintf('    Holdout partition with p=0.33 and call to "subsamples" with boolean indices.\n');
            
            [tr_h,ts_h] = s.partition('holdout',0.33);
            
            s_h1 = s.subsamples(tr_h);
            
            assert(length(s_h1.classes) == 3);
            assert(strcmp(s_h1.classes(1),'1'));
            assert(strcmp(s_h1.classes(2),'2'));
            assert(strcmp(s_h1.classes(2),'2'));
            assert(s_h1.classes_count == 3);
            assert(all(size(s_h1.samples) == [9 100]));
            assert(all(all(s_h1.samples == A_s(tr_h,:))));
            assert(length(s_h1.labels_idx) == 9);
            assert(all(s_h1.labels_idx == c(tr_h)'));
            assert(s_h1.samples_count == 9);
            assert(s_h1.features_count == 100);
            j = 1;
            for i = 1:length(tr_h)
                if tr_h(i)
                    assert(all(all(s_h1.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s_h1.row_count == 10);
            assert(s_h1.col_count == 10);
            assert(all(all(s_h1.samples == gray_images_set.to_samples(s_h1.images))));
            
            s_h2 = s.subsamples(ts_h);
            
            assert(length(s_h2.classes) == 3);
            assert(strcmp(s_h2.classes(1),'1'));
            assert(strcmp(s_h2.classes(2),'2'));
            assert(strcmp(s_h2.classes(2),'2'));
            assert(s_h2.classes_count == 3);
            assert(all(size(s_h2.samples) == [3 100]));
            assert(all(all(s_h2.samples == A_s(ts_h,:))));
            assert(length(s_h2.labels_idx) == 3);
            assert(all(s_h2.labels_idx == c(ts_h)'));
            assert(s_h2.samples_count == 3);
            assert(s_h2.features_count == 100);
            j = 1;
            for i = 1:length(ts_h)
                if ts_h(i)
                    assert(all(all(s_h2.images(:,:,j) == A(:,:,i))));
                    j = j + 1;
                end
            end
            assert(s_h2.row_count == 10);
            assert(s_h2.col_count == 10);
            assert(all(all(s_h2.samples == gray_images_set.to_samples(s_h2.images))));
            
            fprintf('    Call to "subsamples" with natural indices.\n');
            
            s_fi = s.subsamples(1:2:12);
            
            assert(length(s_fi.classes) == 3);
            assert(strcmp(s_fi.classes(1),'1'));
            assert(strcmp(s_fi.classes(2),'2'));
            assert(strcmp(s_fi.classes(3),'3'));
            assert(s_fi.classes_count == 3);
            assert(all(size(s_fi.samples) == [6 100]));
            assert(all(all(s_fi.samples == A_s(1:2:12,:))));
            assert(length(s_fi.labels_idx) == 6);
            assert(all(s_fi.labels_idx == c(1:2:12)'));
            assert(s_fi.samples_count == 6);
            assert(s_fi.features_count == 100);
            j = 1;
            for i = 1:2:12
                assert(all(all(s_fi.images(:,:,j) == A(:,:,i))));
                j = j + 1;
            end
            assert(s_fi.row_count == 10);
            assert(s_fi.col_count == 10);
            assert(all(all(s_fi.samples == gray_images_set.to_samples(s_fi.images))));
            
            fprintf('    Call to "subsamples" with natural indices and redundant selection.\n');
            
            s_fo = s.subsamples([1:12,1:12]);
            
            assert(length(s_fo.classes) == 3);
            assert(strcmp(s_fo.classes(1),'1'));
            assert(strcmp(s_fo.classes(2),'2'));
            assert(strcmp(s_fo.classes(3),'3'));
            assert(s_fo.classes_count == 3);
            assert(all(size(s_fo.samples) == [24 100]));
            assert(all(all(s_fo.samples == [A_s;A_s])));
            assert(length(s_fo.labels_idx) == 24);
            assert(all(s_fo.labels_idx == [c c]'));
            assert(s_fo.samples_count == 24);
            assert(s_fo.features_count == 100);
            j = 1;
            for i = 0:23
                assert(all(all(s_fo.images(:,:,j) == A(:,:,mod(i,12) + 1))));
                j = j + 1;
            end
            assert(s_fo.row_count == 10);
            assert(s_fo.col_count == 10);
            assert(all(all(s_fo.samples == gray_images_set.to_samples(s_fo.images))));
            
            clear all
            
            % Try building from pre-existing data stored in a "samples_set"
            % object. This might be the case when we apply "samples_set"
            % specific transforms to a "gray_images_set".
            
            fprintf('  Function "from_samples".\n');
            
            fprintf('    With mode "none" (default).\n');
            
            A = rand(20,100);
            A_i = zeros(10,10,20);
            for i = 1:20
                A_i(:,:,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            
            s = samples_set({'none'},A,c);
            s_i = gray_images_set.from_samples(s,10,10);
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes(1),'none'));
            assert(s_i.classes_count == 1);
            assert(all(size(s_i.samples) == [20 100]));
            assert(all(all(s_i.samples == A)));
            assert(length(s_i.labels_idx) == 20);
            assert(all(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            for i = 1:20
                assert(all(all(s_i.images(:,:,i) == reshape(A(i,:),[10 10]))));
            end
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == gray_images_set.to_samples(s_i.images))));
            assert(all(all(s.samples == gray_images_set.to_samples(s_i.images))));
            assert(all(all(A == gray_images_set.to_samples(s_i.images))));
            
            figure();
            subplot(1,2,1);
            imshow(utils.format_as_tiles(A_i,4,5));
            title('Original images.');
            subplot(1,2,2);
            imshow(utils.format_as_tiles(s_i.images,4,5));
            title('Images in "gray_images_set".');
            pause(5);
            close(gcf());
            
            clear all;
            
            fprintf('    With mode "none".\n');
            
            A = rand(20,100);
            A_i = zeros(10,10,20);
            for i = 1:20
                A_i(:,:,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            
            s = samples_set({'none'},A,c);
            s_i = gray_images_set.from_samples(s,10,10,'none');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes(1),'none'));
            assert(s_i.classes_count == 1);
            assert(all(size(s_i.samples) == [20 100]));
            assert(all(all(s_i.samples == A)));
            assert(length(s_i.labels_idx) == 20);
            assert(all(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            for i = 1:20
                assert(all(all(s_i.images(:,:,i) == reshape(A(i,:),[10 10]))));
            end
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == gray_images_set.to_samples(s_i.images))));
            assert(all(all(s.samples == gray_images_set.to_samples(s_i.images))));
            assert(all(all(A == gray_images_set.to_samples(s_i.images))));
            
            figure();
            subplot(1,2,1);
            imshow(utils.format_as_tiles(A_i,4,5));
            title('Original images.');
            subplot(1,2,2);
            imshow(utils.format_as_tiles(s_i.images,4,5));
            title('Images in "gray_images_set".');
            pause(5);
            close(gcf());
            
            clear all;
            
            fprintf('    With mode "clamp".\n');
            
            A = 3*rand(20,100) - 1.5;
            A_i = zeros(10,10,20);
            for i = 1:20
                A_i(:,:,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            
            s = samples_set({'none'},A,c);
            s_i = gray_images_set.from_samples(s,10,10,'clamp');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes(1),'none'));
            assert(s_i.classes_count == 1);
            assert(all(size(s_i.samples) == [20 100]));
            assert(length(s_i.labels_idx) == 20);
            assert(all(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            assert(tc.tensor(s_i.images,3) && tc.unitreal(s_i.images));
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == gray_images_set.to_samples(s_i.images))));
            assert(all(all(s_i.samples == gray_images_set.to_samples(s_i.images))));

            figure();
            subplot(1,2,1);
            imshow(utils.format_as_tiles(A_i,4,5,true));
            title('Original images.');
            subplot(1,2,2);
            imshow(utils.format_as_tiles(s_i.images,4,5));
            title('Images in "gray_images_set".');
            pause(5);
            close(gcf());
            
            clear all;
            
            fprintf('    With mode "remap" and "local" (default).\n');
            
            A = 3*rand(20,100) - 1.5;
            A_i = zeros(10,10,20);
            for i = 1:20
                A_i(:,:,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            
            s = samples_set({'none'},A,c);
            s_i = gray_images_set.from_samples(s,10,10,'remap');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes(1),'none'));
            assert(s_i.classes_count == 1);
            assert(all(size(s_i.samples) == [20 100]));
            assert(length(s_i.labels_idx) == 20);
            assert(all(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            assert(tc.tensor(s_i.images,3) && tc.unitreal(s_i.images));
            assert(min(s_i.images(:)) == 0);
            assert(max(s_i.images(:)) == 1);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == gray_images_set.to_samples(s_i.images))));
            assert(all(all(s_i.samples == gray_images_set.to_samples(s_i.images))));
            
            figure();
            subplot(1,2,1);
            imshow(utils.format_as_tiles(A_i,4,5,true));
            title('Original images.');
            subplot(1,2,2);
            imshow(utils.format_as_tiles(s_i.images,4,5));
            title('Images in "gray_images_set".');
            pause(5);
            close(gcf());
            
            clear all;
            
            fprintf('    With mode "remap" and "local".\n');
            
            A = 3*rand(20,100) - 1.5;
            A_i = zeros(10,10,20);
            for i = 1:20
                A_i(:,:,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            
            s = samples_set({'none'},A,c);
            s_i = gray_images_set.from_samples(s,10,10,'remap','local');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes(1),'none'));
            assert(s_i.classes_count == 1);
            assert(all(size(s_i.samples) == [20 100]));
            assert(length(s_i.labels_idx) == 20);
            assert(all(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            assert(tc.tensor(s_i.images,3) && tc.unitreal(s_i.images));
            assert(min(s_i.images(:)) == 0);
            assert(max(s_i.images(:)) == 1);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == gray_images_set.to_samples(s_i.images))));
            assert(all(all(s_i.samples == gray_images_set.to_samples(s_i.images))));
            
            figure();
            subplot(1,2,1);
            imshow(utils.format_as_tiles(A_i,4,5,true));
            title('Original images.');
            subplot(1,2,2);
            imshow(utils.format_as_tiles(s_i.images,4,5));
            title('Images in "gray_images_set".');
            pause(5);
            close(gcf());
            
            clear all;
            
            fprintf('    With mode "remap" and "global".\n');
            
            A = 3*rand(20,100) - 1.5;
            A_i = zeros(10,10,20);
            for i = 1:20
                A_i(:,:,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            
            s = samples_set({'none'},A,c);
            s_i = gray_images_set.from_samples(s,10,10,'remap','global');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes(1),'none'));
            assert(s_i.classes_count == 1);
            assert(all(size(s_i.samples) == [20 100]));
            assert(length(s_i.labels_idx) == 20);
            assert(all(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            assert(tc.tensor(s_i.images,3) && tc.unitreal(s_i.images));
            assert(min(s_i.images(:)) == 0);
            assert(max(s_i.images(:)) == 1);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == gray_images_set.to_samples(s_i.images))));
            assert(all(all(s_i.samples == gray_images_set.to_samples(s_i.images))));
            
            figure();
            subplot(1,2,1);
            imshow(utils.format_as_tiles(A_i,4,5,true));
            title('Original images.');
            subplot(1,2,2);
            imshow(utils.format_as_tiles(s_i.images,4,5));
            title('Images in "gray_images_set".');
            pause(5);
            close(gcf());
            
            clear all;
            
            % Try building from pre-existing data using the "from_data"
            % static method.
            
            fprintf('  Function "from_data".\n');
            
            A = rand(10,10,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,100);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,i),[100 1]);
            end
            
            s = gray_images_set.from_data(A,c);
            
            assert(length(s.classes) == 3);
            assert(strcmp(s.classes(1),'1'));
            assert(strcmp(s.classes(2),'2'));
            assert(strcmp(s.classes(3),'3'));
            assert(s.classes_count == 3);
            assert(all(size(s.samples) == [12 100]));
            assert(all(all(s.samples == A_s)));
            assert(length(s.labels_idx) == 12);
            assert(all(s.labels_idx == c'));
            assert(s.samples_count == 12);
            assert(s.features_count == 100);
            for i = 1:12
                assert(all(all(s.images(:,:,i) == A(:,:,i))));
            end
            assert(s.row_count == 10);
            assert(s.col_count == 10);
            assert(all(all(s.samples == gray_images_set.to_samples(s.images))));
            
            clear all
            
            % Try loading files from a directory. The files are stored in
            % "$PROJECT_ROOT/data/test/". This directory should
            % exist in all distributions of this project.
            
            fprintf('  Function "load_from_dir" with test data.\n');
            
            s = gray_images_set.load_from_dir('../data/test');
            
            assert(length(s.classes) == 1);
            assert(strcmp(s.classes(1),'none'));
            assert(s.classes_count == 1);
            assert(all(size(s.samples) == [7 192*256]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(all(size(s.labels_idx) == [7 1]));
            assert(all(s.labels_idx == ones(7,1)));
            assert(s.samples_count == 7);
            assert(s.features_count == 192*256);
            assert(tc.tensor(s.images,3) && tc.unitreal(s.images));
            assert(s.row_count == 192);
            assert(s.col_count == 256);
            assert(all(all(s.samples == gray_images_set.to_samples(s.images))));
            
            clear all
            
            % Try loading files from a directory, with forced size. The 
            % files are stored in "$PROJECT_ROOT/data/test/". This 
            % directory should exist in all distributions of this project.
            
            fprintf('  Function "load_from_dir" with forced size on test data.\n');
            
            s = gray_images_set.load_from_dir('../data/test',[96 128]);
            
            assert(length(s.classes) == 1);
            assert(strcmp(s.classes(1),'none'));
            assert(s.classes_count == 1);
            assert(all(size(s.samples) == [7 96*128]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(all(size(s.labels_idx) == [7 1]));
            assert(all(s.labels_idx == ones(7,1)));
            assert(s.samples_count == 7);
            assert(s.features_count == 96*128);
            assert(tc.tensor(s.images,3) && tc.unitreal(s.images));
            assert(s.row_count == 96);
            assert(s.col_count == 128);
            assert(all(all(s.samples == gray_images_set.to_samples(s.images))));
            
            clear all
            
            % Try loading files from a directory which contains other types
            % of files besides images. The files are stored in
            % "$PROJECT_ROOT/data/test/heterogeneous_dir". This directory
            % should exits for all distributions of this project.
            
            fprintf('  Function "load_from_dir" with heterogenous directory.\n');
            
            s = gray_images_set.load_from_dir('../data/test/heterogeneous_dir');
            
            assert(length(s.classes) == 1);
            assert(strcmp(s.classes(1),'none'));
            assert(s.classes_count == 1);
            assert(all(size(s.samples) == [2 192*256]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(all(size(s.labels_idx) == [2 1]));
            assert(all(s.labels_idx == ones(2,1)));
            assert(s.samples_count == 2);
            assert(s.features_count == 192*256);
            assert(tc.tensor(s.images,3) && tc.unitreal(s.images));
            assert(s.row_count == 192);
            assert(s.col_count == 256);
            assert(all(all(s.samples == gray_images_set.to_samples(s.images))));
            
            clear all
            
            % Try some invalid calls to "load_from_dir". These test the
            % failure modes of this function. We're interested in things
            % beyond the caller's control, like empty directories,
            % insufficient access rights and images of different sizes.
            
            fprintf('  Function "load_from_dir" with invalid external inputs.\n');
            
            try
                s = gray_images_set.load_from_dir('../data/test_aaa');
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
                s = gray_images_set.load_from_dir('../data/test');
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
                s = gray_images_set.load_from_dir('../data/test/empty_dir');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('    Passes "Empty directory!"\n');
                else
                    assert(false);
                end
            end
            
            clear all
            
            % Try loading images saved in the MNIST format. The images are
            % stored in "$PROJECT_ROOT/data/mnist/train-images-idx3-ubyte",
            % while the labels are stored in
            % "$PROJECT_ROOT/data/mnist/train-labels-idx1-ubyte". These two
            % files should exist in all distributions of this project.
            
            fprintf('  Function "load_mnist" with MNIST training data.\n');
            
            s = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte');
            
            assert(length(s.classes) == 10);
            assert(strcmp(s.classes(1),'0'));
            assert(strcmp(s.classes(2),'1'));
            assert(strcmp(s.classes(3),'2'));
            assert(strcmp(s.classes(4),'3'));
            assert(strcmp(s.classes(5),'4'));
            assert(strcmp(s.classes(6),'5'));
            assert(strcmp(s.classes(7),'6'));
            assert(strcmp(s.classes(8),'7'));
            assert(strcmp(s.classes(9),'8'));
            assert(strcmp(s.classes(10),'9'));
            assert(s.classes_count == 10);
            assert(all(size(s.samples) == [60000 28*28]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(all(size(s.labels_idx) == [60000 1]));
            assert(tc.vector(s.labels_idx) && tc.labels_idx(s.labels_idx,s.classes));
            assert(s.samples_count == 60000);
            assert(s.features_count == 28*28);
            assert(tc.tensor(s.images,3) && tc.unitreal(s.images));
            assert(s.row_count == 28);
            assert(s.col_count == 28);
            assert(all(all(s.samples == gray_images_set.to_samples(s.images))));
            
            clear all
            
            % Try some invalid calls to "load_mnist". These test the
            % failure modes of this function. We're interested in things
            % beyond the caller's control, like improper access rights to
            % input files or badly formatted files. We will not test the
            % latter error path in detail though, as it is highly
            % improbable given our current setup. Maybe later, when other
            % MNIST like datasets are added, we should test what happens if
            % not enough images are present or any of a number of binary
            % format decoding errors.
            
            fprintf('  Function "load_mnist" with invalid external inputs.\n');
            
            try
                s = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte_aaa','../data/mnist/train-labels-idx1-ubyte');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not load images in "../data/mnist/train-images-idx3-ubyte_aaa": No such file or directory!')
                    fprintf('    Passes "No such file or directory!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte_aaa');
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
                s = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte');
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
                s = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/train-labels-idx1-ubyte');
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
                s = gray_images_set.load_mnist('../data/test/scenes_small1.jpg','../data/mnist/train-labels-idx1-ubyte');
            catch exp
                if strcmp(exp.message,'Images file "../data/test/scenes_small1.jpg" not in MNIST format!')
                    fprintf('    Passes "Not in MNIST format!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/test/scenes_small1.jpg');
            catch exp
                if strcmp(exp.message,'Labels file "../data/test/scenes_small1.jpg" not in MNIST format!')
                    fprintf('    Passes "Not in MNIST format!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s = gray_images_set.load_mnist('../data/mnist/train-images-idx3-ubyte','../data/mnist/t10k-labels-idx1-ubyte');
            catch exp
                if strcmp(exp.message,'Different number of labels in "../data/mnist/t10k-labels-idx1-ubyte" for images in "../data/mnist/train-images-idx3-ubyte"!')
                    fprintf('    Passes "Different number of labels!".\n');
                else
                    assert(false);
                end
            end
            
            clear all;
        end
    end
end
