classdef image < dataset
    properties (GetAccess=public,SetAccess=immutable)
        images;
        layers_count;
        row_count;
        col_count;
    end
    
    methods (Access=public)
        function [obj] = image(classes,images,labels_idx)
            assert(tc.vector(classes) && tc.labels(classes));
            assert(tc.tensor(images,4) && tc.unitreal(images));
            assert(tc.vector(labels_idx) && tc.match_dims(images,labels_idx,4) && ...
                   tc.labels_idx(labels_idx,classes));
               
            obj = obj@dataset(classes,datasets.image.to_samples(images),labels_idx);
            obj.images = images;
            obj.layers_count = size(images,3);
            obj.row_count = size(images,1);
            obj.col_count = size(images,2);
        end
        
        function [o] = eq(obj,another_image)
            assert(tc.scalar(obj) && tc.datasets_image(obj));
            assert(tc.scalar(another_image) && tc.datasets_image(another_image));
            
            o = true;
            o = o && obj.eq@dataset(another_image);
            o = o && tc.check(size(obj.images) == size(another_image.images));
            o = o && tc.check(obj.images == another_image.images);
            o = o && (obj.layers_count == another_image.layers_count);
            o = o && (obj.row_count == another_image.row_count);
            o = o && (obj.col_count == another_image.col_count);
        end
        
        function [o] = compatible(obj,another_image)
            assert(tc.scalar(obj) && tc.datasets_image(obj));
            assert(tc.scalar(another_image) && tc.datasets_image(another_image));
            
            o = true;
            o = o && obj.compatible@dataset(another_image);
            o = o && (obj.layers_count == another_image.layers_count);
            o = o && (obj.row_count == another_image.row_count);
            o = o && (obj.col_count == another_image.col_count);
        end
        
        function [new_image] = subsamples(obj,index)
            assert(tc.scalar(obj) && tc.datasets_image(obj));
            assert(tc.vector(index) && ...
                   ((tc.logical(index) && tc.match_dims(obj.samples,index,1)) || ...
                    (tc.natural(index) && tc.check(index > 0 & index <= obj.samples_count))));
            
            new_image = datasets.image(obj.classes,obj.images(:,:,:,index),obj.labels_idx(index));
        end
    end
    
    methods (Static,Access=public)        
        function [new_image] = from_data(images,labels)
            assert(tc.tensor(images,4) && tc.unitreal(images));
            assert(tc.vector(labels) && tc.match_dims(images,labels,4) && tc.labels(labels));
            
            [labels_idx_t,classes_t] = grp2idx(labels);
            new_image = datasets.image(classes_t,images,labels_idx_t);
        end
        
        function [new_image] = from_fulldata(classes,images,labels_idx)
            assert(tc.vector(classes) && tc.labels(classes));
            assert(tc.tensor(images,4) && tc.unitreal(images));
            assert(tc.vector(labels_idx) && tc.match_dims(images,labels_idx,4) && ...
                   tc.labels_idx(labels_idx,classes));
               
            new_image = datasets.image(classes,images,labels_idx);
        end
        
        function [new_image] = from_dataset(dataset_d,layers_count,row_count,col_count,remap_type,remap_mode)
            assert(tc.scalar(dataset_d) && tc.dataset(dataset_d));
            assert(tc.scalar(layers_count) && tc.natural(layers_count) && (layers_count > 0));
            assert(tc.scalar(row_count) && tc.natural(row_count) && (row_count > 0));
            assert(tc.scalar(col_count) && tc.natural(col_count) && (col_count > 0));
            assert(~exist('remap_type','var') || (tc.scalar(remap_type) && tc.string(remap_type) && ...
                    (strcmp(remap_type,'none') || strcmp(remap_type,'clamp') || strcmp(remap_type,'remap'))));
            assert(~(exist('remap_type','var') && exist('remap_mode','var')) || ...
                    (strcmp(remap_type,'remap') && (tc.scalar(remap_mode) && tc.string(remap_mode) && ...
                                                     (strcmp(remap_mode,'local') || strcmp(remap_mode,'global')))));
                
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
            
            new_images = zeros(row_count,col_count,layers_count,dataset_d.samples_count);
            
            for i = 1:dataset_d.samples_count
                new_images(:,:,:,i) = reshape(dataset_d.samples(i,:),[row_count col_count layers_count]);
            end
            
            if strcmp(remap_type_t,'none')
                new_images = new_images;
            elseif strcmp(remap_type_t,'clamp')
                new_images = utils.clamp_images_to_unit(new_images);
            else
                new_images = utils.remap_images_to_unit(new_images,remap_mode_t);
            end
            
            new_image = datasets.image(dataset_d.classes,new_images,dataset_d.labels_idx);
        end
        
        function [new_image] = load_from_dir(images_dir_path,mode,force_size)
            assert(tc.scalar(images_dir_path) && tc.string(images_dir_path));
            assert(~exist('mode','var') || ...
                    (tc.scalar(mode) && tc.string(mode) && ...
                      (strcmp(mode,'gray') || strcmp(mode,'color'))));
            assert(~exist('force_size','var') || ...
                    (tc.vector(force_size) && (length(force_size) == 2) && ...
                      tc.natural(force_size) && tc.check(force_size > 1)));
                  
            if exist('mode','var')
                mode_t = mode;
            else
                mode_t = 'gray';
            end               
               
            paths = dir(images_dir_path);
            images_t = [];
            current_image = 1;
            
            for i = 1:length(paths)
                try
                    image = imread(fullfile(images_dir_path,paths(i).name));
                    
                    if strcmp(mode_t,'gray')
                        image = double(rgb2gray(image)) ./ 255;
                    else
                        image = double(image) ./ 255;
                    end
                    
                    if exist('force_size','var')
                        image = imresize(image,force_size);
                        
                        % Correct small domain overflows caused by resizing.
                        image = utils.clamp_images_to_unit(image);
                    end
                    
                    if (current_image > 1) && ...
                        (~tc.check(size(image) == size(images_t(:,:,:,1))))
                        throw(MException('master:datasets:image:load_from_dir:NoLoad',...
                                         'Images are of different sizes!'));
                    end

                    images_t(:,:,:,current_image) = image;
                    current_image = current_image + 1;
                catch exp
                    if isempty(regexp(exp.identifier,'MATLAB:imread:.*','ONCE'))
                        throw(MException('master:datasets:image:load_from_dir:NoLoad',exp.message));
                    end
                end
            end
            
            if isempty(images_t)
                throw(MException('master:datasets:image:load_from_dir:NoLoad',...
                                 'Could not find any acceptable images in the directory.'));
            end
            
            new_image = datasets.image({'none'},images_t,ones(current_image - 1,1));
        end
        
        function [new_image] = load_mnist(images_path,labels_path)
            assert(tc.scalar(images_path) && tc.string(images_path));
            assert(tc.scalar(labels_path) && tc.string(labels_path));
            
            [images_fid,images_msg] = fopen(images_path,'rb');
            
            if images_fid == -1
                throw(MException('master:datasets:image:load_mnist:NoLoad',...
                    sprintf('Could not load images in "%s": %s!',images_path,images_msg)))
            end
            
            [labels_fid,labels_msg] = fopen(labels_path,'rb');
            
            if labels_fid == -1
                fclose(images_fid);
                throw(MException('master:datasets:image:load_mnist:NoLoad',...
                    sprintf('Could not load labels in "%s": %s!',labels_path,labels_msg)))
            end
            
            try
                images_magic = datasets.image.high2low(fread(images_fid,4,'uint8=>uint32'));
                
                if images_magic ~= 2051
                    throw(MException('master:datasets:image:load_mnist:NoLoad',...
                        sprintf('Images file "%s" not in MNIST format!',images_path)));
                end
                
                labels_magic = datasets.image.high2low(fread(labels_fid,4,'uint8=>uint32'));
                
                if labels_magic ~= 2049
                    throw(MException('master:datasets:image:load_mnist:NoLoad',...
                        sprintf('Labels file "%s" not in MNIST format!',labels_path)));
                end
                
                images_count = datasets.image.high2low(fread(images_fid,4,'uint8=>uint32'));
                labels_count = datasets.image.high2low(fread(labels_fid,4,'uint8=>uint32'));
                
                if images_count ~= labels_count
                    throw(MException('master:datasets:image:load_mnist:NoLoad',...
                        sprintf('Different number of labels in "%s" for images in "%s"!',labels_path,images_path)));
                end
                
                row_count_t = datasets.image.high2low(fread(images_fid,4,'uint8=>uint32'));
                col_count_t = datasets.image.high2low(fread(images_fid,4,'uint8=>uint32'));
                
                images_t = zeros(row_count_t,col_count_t,1,images_count);
                
                for i = 1:images_count
                    images_t(:,:,1,i) = fread(images_fid,[row_count_t col_count_t],'uint8=>double')' ./ 255;
                end
                
                labels_t = fread(labels_fid,[images_count 1],'uint8');
                
                fclose(images_fid);
                fclose(labels_fid);
            catch exp
                fclose(images_fid);
                fclose(labels_fid);
                throw(MException('master:datasets:image:load_mnist:NoLoad',exp.message));
            end

            new_image = datasets.image.from_data(images_t,labels_t);
        end
    end
    
    methods (Static,Access=protected)
        function [samples] = to_samples(images)
            features_count = size(images,1) * size(images,2) * size(images,3);
            samples = zeros(size(images,4),features_count);
            
            for i = 1:size(images,4)
                samples(i,:) = reshape(images(:,:,:,i),[1 features_count]);
            end
        end
        
        function [out] = high2low(bytes)
            out = bitshift(bytes(4),0) + bitshift(bytes(3),8) + ...
                  bitshift(bytes(2),16) + bitshift(bytes(1),24);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "datasets.image".\n');
            
            fprintf('  Proper construction.\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
            
            s = datasets.image({'1' '2' '3'},A,c);
            
            assert(length(s.classes) == 3);
            assert(strcmp(s.classes{1},'1'));
            assert(strcmp(s.classes{2},'2'));
            assert(strcmp(s.classes{3},'3'));
            assert(s.classes_count == 3);
            assert(tc.check(s.samples == A_s));
            assert(tc.check(s.labels_idx == c'));
            assert(s.samples_count == 12);
            assert(s.features_count == 300);
            assert(tc.check(s.images == A));
            assert(s.layers_count == 3);
            assert(s.row_count == 10);
            assert(s.col_count == 10);
            assert(all(all(s.samples == datasets.image.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('  Functions "eq" and "ne".\n');
            
            s1 = datasets.image({'1' '2'},cat(4,cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s2 = datasets.image({'1' '2'},cat(4,cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s3 = datasets.image({'1' '2' '3'},cat(4,cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                    cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s4 = datasets.image({'true' 'false'},cat(4,cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                       cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s5 = datasets.image([1 2],cat(4,cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                            cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s6 = datasets.image([true false],cat(4,cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                   cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s7 = datasets.image({'1' '2'},cat(4,cat(3,[0.1 0.2 0.3; 0.1 0.3 0.2],[0.1 0.2 0.4; 0.1 0.4 0.2]),...
                                                cat(3,[0.1 0.2 0.3; 0.1 0.3 0.2],[0.1 0.2 0.4; 0.1 0.4 0.2])),[1 2]);
            s8 = datasets.image({'1' '2'},cat(4,cat(3,[0.1 0.2 0.2 0.1],[0.1 0.3 0.3 0.1]),...
                                                cat(3,[0.1 0.2 0.2 0.1],[0.1 0.3 0.3 0.1])),[1 2]);
            s9 = datasets.image({'1' '2'},cat(4,cat(3,[0.1 0.2; 0.2 0.2],[0.1 0.3; 0.3 0.1]),...
                                                cat(3,[0.1 0.2; 0.2 0.2],[0.1 0.3; 0.3 0.1])),[1 2]);
            s10 = datasets.image({'1' '2'},cat(4,cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                 cat(3,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 1]);
            
            assert(s1 == s2);
            assert(s1 ~= s3);
            assert(s1 ~= s4);
            assert(s1 ~= s5);
            assert(s1 ~= s6);
            assert(s1 ~= s7);
            assert(s1 ~= s8);
            assert(s1 ~= s9);
            assert(s1 ~= s10);
            
            clearvars -except display;
            
            fprintf('  Function "compatible".\n');
            
            s1 = datasets.image({'1' '2'},rand(10,10,3,3),randi(2,3,1));
            s2 = datasets.image({'1' '2'},rand(10,10,3,4),randi(2,4,1));
            s3 = datasets.image({'1' '2' '3'},rand(10,10,3,3),randi(2,3,1));
            s4 = datasets.image({'hello' 'world'},rand(10,10,3,3),randi(2,3,1));
            s5 = datasets.image([1 2],rand(10,10,3,3),randi(2,3,1));
            s6 = datasets.image([true false],rand(10,10,3,3),randi(2,3,1));
            s7 = datasets.image({'1' '2'},rand(12,10,3,3),randi(2,3,1));
            s8 = datasets.image({'1' '2'},rand(10,12,3,3),randi(2,3,1));
            s9 = datasets.image({'1' '2'},rand(10,10,4,3),randi(2,3,1));
            s10 = datasets.image({'1' '2'},rand(4,25,3,4),randi(2,4,1));
            
            assert(s1.compatible(s2) == true);
            assert(s1.compatible(s3) == false);
            assert(s1.compatible(s4) == false);
            assert(s1.compatible(s5) == false);
            assert(s1.compatible(s6) == false);
            assert(s1.compatible(s7) == false);
            assert(s1.compatible(s8) == false);
            assert(s1.compatible(s9) == false);
            assert(s1.compatible(s10) == false);
            
            clearvars -except display;
            
            fprintf('  Functions "partition" and "subsamples".\n');
            
            fprintf('    2-fold partition and call to "subsamples" with boolean indices.\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
                        
            s = datasets.image({'1' '2' '3'},A,c);
            
            [tr_f,ts_f] = s.partition('kfold',2);
            
            s_f11 = s.subsamples(tr_f(:,1));
            
            assert(length(s_f11.classes) == 3);
            assert(strcmp(s_f11.classes{1},'1'));
            assert(strcmp(s_f11.classes{2},'2'));
            assert(strcmp(s_f11.classes{3},'3'));
            assert(s_f11.classes_count == 3);
            assert(tc.check(s_f11.samples == A_s(tr_f(:,1),:)));
            assert(tc.check(s_f11.labels_idx == c(tr_f(:,1))'));
            assert(s_f11.samples_count == 6);
            assert(s_f11.features_count == 300);
            assert(tc.check(s_f11.images == A(:,:,:,tr_f(:,1))));
            assert(s_f11.layers_count == 3);
            assert(s_f11.row_count == 10);
            assert(s_f11.col_count == 10);
            assert(all(all(s_f11.samples == datasets.image.to_samples(s_f11.images))));
            
            s_f12 = s.subsamples(ts_f(:,1));
            
            assert(length(s_f12.classes) == 3);
            assert(strcmp(s_f12.classes{1},'1'));
            assert(strcmp(s_f12.classes{2},'2'));
            assert(strcmp(s_f12.classes{3},'3'));
            assert(s_f12.classes_count == 3);
            assert(tc.check(s_f12.samples == A_s(ts_f(:,1),:)));
            assert(tc.check(s_f12.labels_idx == c(ts_f(:,1))'));
            assert(s_f12.samples_count == 6);
            assert(s_f12.features_count == 300);
            assert(tc.check(s_f12.images == A(:,:,:,ts_f(:,1))));
            assert(s_f12.layers_count == 3);
            assert(s_f12.row_count == 10);
            assert(s_f12.col_count == 10);
            assert(all(all(s_f12.samples == datasets.image.to_samples(s_f12.images))));
            
            s_f21 = s.subsamples(tr_f(:,2));

            assert(length(s_f21.classes) == 3);
            assert(strcmp(s_f21.classes{1},'1'));
            assert(strcmp(s_f21.classes{2},'2'));
            assert(strcmp(s_f21.classes{3},'3'));
            assert(s_f21.classes_count == 3);
            assert(tc.check(s_f21.samples == A_s(tr_f(:,2),:)));
            assert(length(s_f21.labels_idx) == 6);
            assert(all(s_f21.labels_idx == c(tr_f(:,2))'));
            assert(s_f21.samples_count == 6);
            assert(s_f21.features_count == 300);
            assert(tc.check(s_f21.images == A(:,:,:,tr_f(:,2))));
            assert(s_f21.layers_count == 3);
            assert(s_f21.row_count == 10);
            assert(s_f21.col_count == 10);
            assert(all(all(s_f21.samples == datasets.image.to_samples(s_f21.images))));
            
            s_f22 = s.subsamples(ts_f(:,2));
            
            assert(length(s_f22.classes) == 3);
            assert(strcmp(s_f22.classes{1},'1'));
            assert(strcmp(s_f22.classes{2},'2'));
            assert(strcmp(s_f22.classes{3},'3'));
            assert(s_f22.classes_count == 3);
            assert(tc.check(s_f22.samples == A_s(ts_f(:,2),:)));
            assert(tc.check(s_f22.labels_idx == c(ts_f(:,2))'));
            assert(s_f22.samples_count == 6);
            assert(s_f22.features_count == 300);
            assert(tc.check(s_f22.images == A(:,:,:,ts_f(:,2))));
            assert(s_f22.layers_count == 3);
            assert(s_f22.row_count == 10);
            assert(s_f22.col_count == 10);
            assert(all(all(s_f22.samples == datasets.image.to_samples(s_f22.images))));
            
            clearvars -except display;
            
            fprintf('    Holdout partition with p=0.33 and call to "subsamples" with boolean indices.\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
                        
            s = datasets.image({'1' '2' '3'},A,c);
            
            [tr_h,ts_h] = s.partition('holdout',0.33);
            
            s_h1 = s.subsamples(tr_h);
            
            assert(length(s_h1.classes) == 3);
            assert(strcmp(s_h1.classes{1},'1'));
            assert(strcmp(s_h1.classes{2},'2'));
            assert(strcmp(s_h1.classes{2},'2'));
            assert(s_h1.classes_count == 3);
            assert(tc.check(s_h1.samples == A_s(tr_h,:)));
            assert(tc.check(s_h1.labels_idx == c(tr_h)'));
            assert(s_h1.samples_count == 9);
            assert(s_h1.features_count == 300);
            assert(tc.check(s_h1.images == A(:,:,:,tr_h)));
            assert(s_h1.layers_count == 3);
            assert(s_h1.row_count == 10);
            assert(s_h1.col_count == 10);
            assert(all(all(s_h1.samples == datasets.image.to_samples(s_h1.images))));
            
            s_h2 = s.subsamples(ts_h);
            
            assert(length(s_h2.classes) == 3);
            assert(strcmp(s_h2.classes{1},'1'));
            assert(strcmp(s_h2.classes{2},'2'));
            assert(strcmp(s_h2.classes{2},'2'));
            assert(s_h2.classes_count == 3);
            assert(tc.check(s_h2.samples == A_s(ts_h,:)));
            assert(tc.check(s_h2.labels_idx == c(ts_h)'));
            assert(s_h2.samples_count == 3);
            assert(s_h2.features_count == 300);
            assert(tc.check(s_h2.images == A(:,:,:,ts_h)));
            assert(s_h2.layers_count == 3);
            assert(s_h2.row_count == 10);
            assert(s_h2.col_count == 10);
            assert(all(all(s_h2.samples == datasets.image.to_samples(s_h2.images))));
            
            clearvars -except display;
            
            fprintf('    Call to "subsamples" with natural indices.\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
                        
            s = datasets.image({'1' '2' '3'},A,c);
            
            s_fi = s.subsamples(1:2:12);
            
            assert(length(s_fi.classes) == 3);
            assert(strcmp(s_fi.classes{1},'1'));
            assert(strcmp(s_fi.classes{2},'2'));
            assert(strcmp(s_fi.classes{3},'3'));
            assert(s_fi.classes_count == 3);
            assert(tc.check(s_fi.samples == A_s(1:2:12,:)));
            assert(tc.check(s_fi.labels_idx == c(1:2:12)'));
            assert(s_fi.samples_count == 6);
            assert(s_fi.features_count == 300);
            assert(tc.check(s_fi.images == A(:,:,:,1:2:12)));
            assert(s_fi.layers_count == 3);
            assert(s_fi.row_count == 10);
            assert(s_fi.col_count == 10);
            assert(all(all(s_fi.samples == datasets.image.to_samples(s_fi.images))));
            
            clearvars -except display;
            
            fprintf('    Call to "subsamples" with natural indices and redundant selection.\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
                        
            s = datasets.image({'1' '2' '3'},A,c);
            
            s_fo = s.subsamples([1:12,1:12]);
            
            assert(length(s_fo.classes) == 3);
            assert(strcmp(s_fo.classes{1},'1'));
            assert(strcmp(s_fo.classes{2},'2'));
            assert(strcmp(s_fo.classes{3},'3'));
            assert(s_fo.classes_count == 3);
            assert(tc.check(s_fo.samples == [A_s;A_s]));
            assert(tc.check(s_fo.labels_idx == [c c]'));
            assert(s_fo.samples_count == 24);
            assert(s_fo.features_count == 300);
            assert(tc.check(s_fo.images == cat(4,A,A)));
            assert(s_fo.layers_count == 3);
            assert(s_fo.row_count == 10);
            assert(s_fo.col_count == 10);
            assert(all(all(s_fo.samples == datasets.image.to_samples(s_fo.images))));
            
            clearvars -except display;
            
            fprintf('  Function "from_data".\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
            
            s = datasets.image.from_data(A,c);
            
            assert(length(s.classes) == 3);
            assert(strcmp(s.classes{1},'1'));
            assert(strcmp(s.classes{2},'2'));
            assert(strcmp(s.classes{3},'3'));
            assert(s.classes_count == 3);
            assert(tc.check(s.samples == A_s));
            assert(tc.check(s.labels_idx == c'));
            assert(s.samples_count == 12);
            assert(s.features_count == 300);
            assert(tc.check(s.images == A));
            assert(s.layers_count == 3);
            assert(s.row_count == 10);
            assert(s.col_count == 10);
            assert(all(all(s.samples == datasets.image.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('  Function "from_fulldata".\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
            
            s = datasets.image.from_fulldata({'1' '2' '3'},A,c);
            
            assert(tc.datasets_image(s));
            assert(length(s.classes) == 3);
            assert(strcmp(s.classes{1},'1'));
            assert(strcmp(s.classes{2},'2'));
            assert(strcmp(s.classes{3},'3'));
            assert(s.classes_count == 3);
            assert(tc.check(s.samples == A_s));
            assert(tc.check(s.labels_idx == c'));
            assert(s.samples_count == 12);
            assert(s.features_count == 300);
            assert(tc.check(s.images == A));
            assert(s.layers_count == 3);
            assert(s.row_count == 10);
            assert(s.col_count == 10);
            assert(all(all(s.samples == datasets.image.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('  Function "from_dataset".\n');
            
            fprintf('    Single layer and mode "none" (default).\n');
            
            A = rand(20,100);
            A_i = zeros(10,10,1,20);
            for i = 1:20
                A_i(:,:,1,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,1,10,10);
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            assert(tc.check(s_i.images == A_i));
            assert(s_i.layers_count == 1);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(s.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Single layer and mode "none".\n');
            
            A = rand(20,100);
            A_i = zeros(10,10,1,20);
            for i = 1:20
                A_i(:,:,1,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,1,10,10,'none');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            assert(tc.check(s_i.images == A_i));
            assert(s_i.layers_count == 1);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(s.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Single layer and mode "clamp".\n');
            
            A = 3*rand(20,100) - 1.5;
            A_i = zeros(10,10,1,20);
            for i = 1:20
                A_i(:,:,1,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            A_ii = utils.clamp_images_to_unit(A_i);
            A_is = datasets.image.to_samples(A_ii);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,1,10,10,'clamp');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A_is));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            assert(tc.check(s_i.images == A_ii));
            assert(s_i.layers_count == 1);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A_is == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5,true));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Single layer and mode "remap" and "local" (default).\n');
            
            A = 3*rand(20,100) - 1.5;
            A_i = zeros(10,10,1,20);
            for i = 1:20
                A_i(:,:,1,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            A_ii = utils.remap_images_to_unit(A_i);
            A_is = datasets.image.to_samples(A_ii);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,1,10,10,'remap');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A_is));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            assert(tc.check(s_i.images == A_ii));
            assert(s_i.layers_count == 1);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A_is == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5,true));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Single layer and mode "remap" and "local".\n');
            
            A = 3*rand(20,100) - 1.5;
            A_i = zeros(10,10,1,20);
            for i = 1:20
                A_i(:,:,1,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            A_ii = utils.remap_images_to_unit(A_i,'local');
            A_is = datasets.image.to_samples(A_ii);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,1,10,10,'remap','local');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A_is));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            assert(tc.check(s_i.images == A_ii));
            assert(s_i.layers_count == 1);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A_is == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5,true));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Single layer and mode "remap" and "global".\n');
            
            A = 3*rand(20,100) - 1.5;
            A(5:8,:) = 2*A(5:8,:);
            A_i = zeros(10,10,1,20);
            for i = 1:20
                A_i(:,:,1,i) = reshape(A(i,:),[10 10]);
            end
            c = ones(20,1);
            A_ii = utils.remap_images_to_unit(A_i,'global');
            A_is = datasets.image.to_samples(A_ii);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,1,10,10,'remap','global');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A_is));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 100);
            assert(tc.check(s_i.images == A_ii));
            assert(s_i.layers_count == 1);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A_is == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5,true));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Three layers and mode "none" (default).\n');
            
            A = rand(20,300);
            A_i = zeros(10,10,3,20);
            for i = 1:20
                A_i(:,:,:,i) = reshape(A(i,:),[10 10 3]);
            end
            c = ones(20,1);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,3,10,10);
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 300);
            assert(tc.check(s_i.images == A_i));
            assert(s_i.layers_count == 3);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(s.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Three layers and mode "none".\n');
            
            A = rand(20,300);
            A_i = zeros(10,10,3,20);
            for i = 1:20
                A_i(:,:,:,i) = reshape(A(i,:),[10 10 3]);
            end
            c = ones(20,1);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,3,10,10,'none');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 300);
            assert(tc.check(s_i.images == A_i));
            assert(s_i.layers_count == 3);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(s.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Three layers and mode "clamp".\n');
            
            A = 3*rand(20,300) - 1.5;
            A_i = zeros(10,10,3,20);
            for i = 1:20
                A_i(:,:,:,i) = reshape(A(i,:),[10 10 3]);
            end
            c = ones(20,1);
            A_ii = utils.clamp_images_to_unit(A_i);
            A_is = datasets.image.to_samples(A_ii);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,3,10,10,'clamp');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A_is));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 300);
            assert(tc.check(s_i.images == A_ii));
            assert(s_i.layers_count == 3);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A_is == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5,true));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Three layers and mode "remap" and "local" (default).\n');
            
            A = 3*rand(20,300) - 1.5;
            A_i = zeros(10,10,3,20);
            for i = 1:20
                A_i(:,:,:,i) = reshape(A(i,:),[10 10 3]);
            end
            c = ones(20,1);
            A_ii = utils.remap_images_to_unit(A_i);
            A_is = datasets.image.to_samples(A_ii);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,3,10,10,'remap');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A_is));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 300);
            assert(tc.check(s_i.images == A_ii));
            assert(s_i.layers_count == 3);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A_is == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5,true));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Three layers and mode "remap" and "local".\n');
            
            A = 3*rand(20,300) - 1.5;
            A_i = zeros(10,10,3,20);
            for i = 1:20
                A_i(:,:,:,i) = reshape(A(i,:),[10 10 3]);
            end
            c = ones(20,1);
            A_ii = utils.remap_images_to_unit(A_i,'local');
            A_is = datasets.image.to_samples(A_ii);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,3,10,10,'remap','local');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A_is));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 300);
            assert(tc.check(s_i.images == A_ii));
            assert(s_i.layers_count == 3);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A_is == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5,true));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('    Three layers and mode "remap" and "global".\n');
            
            A = 3*rand(20,300) - 1.5;
            A(5:8,:) = 2*A(5:8,:);
            A_i = zeros(10,10,3,20);
            for i = 1:20
                A_i(:,:,:,i) = reshape(A(i,:),[10 10 3]);
            end
            c = ones(20,1);
            A_ii = utils.remap_images_to_unit(A_i,'global');
            A_is = datasets.image.to_samples(A_ii);
            
            s = dataset({'none'},A,c);
            s_i = datasets.image.from_dataset(s,3,10,10,'remap','global');
            
            assert(length(s_i.classes) == 1);
            assert(strcmp(s_i.classes{1},'none'));
            assert(s_i.classes_count == 1);
            assert(tc.check(s_i.samples == A_is));
            assert(tc.check(s_i.labels_idx == c));
            assert(s_i.samples_count == 20);
            assert(s_i.features_count == 300);
            assert(tc.check(s_i.images == A_ii));
            assert(s_i.layers_count == 3);
            assert(s_i.row_count == 10);
            assert(s_i.col_count == 10);
            assert(all(all(s_i.samples == datasets.image.to_samples(s_i.images))));
            assert(all(all(A_is == datasets.image.to_samples(s_i.images))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(A_i,4,5,true));
                title('Original images.');
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_i.images,4,5));
                title('Images in "datasets.image".');
                pause(5);
                close(gcf());
            end
            
            clearvars -except display;
            
            fprintf('  Function "load_from_dir".\n');
            
            fprintf('    With mode "gray" (default) and file size.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small');
            
            assert(length(s.classes) == 1);
            assert(strcmp(s.classes{1},'none'));
            assert(s.classes_count == 1);
            assert(all(size(s.samples) == [7 192*256]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(tc.check(s.labels_idx == ones(7,1)));
            assert(s.samples_count == 7);
            assert(s.features_count == 192*256);
            assert(tc.check(size(s.images) == [192 256 1 7]));
            assert(tc.tensor(s.images,4) && tc.unitreal(s.images));
            assert(s.layers_count == 1);
            assert(s.row_count == 192);
            assert(s.col_count == 256);
            assert(all(all(s.samples == datasets.image.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('    With mode "gray" and file size.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','gray');
            
            assert(length(s.classes) == 1);
            assert(strcmp(s.classes{1},'none'));
            assert(s.classes_count == 1);
            assert(all(size(s.samples) == [7 192*256]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(tc.check(s.labels_idx == ones(7,1)));
            assert(s.samples_count == 7);
            assert(s.features_count == 192*256);
            assert(tc.check(size(s.images) == [192 256 1 7]));
            assert(tc.tensor(s.images,4) && tc.unitreal(s.images));
            assert(s.layers_count == 1);
            assert(s.row_count == 192);
            assert(s.col_count == 256);
            assert(all(all(s.samples == datasets.image.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('    With mode "color" and file size.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','color');
            
            assert(length(s.classes) == 1);
            assert(strcmp(s.classes{1},'none'));
            assert(s.classes_count == 1);
            assert(all(size(s.samples) == [7 3*192*256]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(tc.check(s.labels_idx == ones(7,1)));
            assert(s.samples_count == 7);
            assert(s.features_count == 3*192*256);
            assert(tc.check(size(s.images) == [192 256 3 7]));
            assert(tc.tensor(s.images,4) && tc.unitreal(s.images));
            assert(s.layers_count == 3);
            assert(s.row_count == 192);
            assert(s.col_count == 256);
            assert(all(all(s.samples == datasets.image.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('    With mode "gray" and forced size.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','gray',[96 128]);
            
            assert(length(s.classes) == 1);
            assert(strcmp(s.classes{1},'none'));
            assert(s.classes_count == 1);
            assert(all(size(s.samples) == [7 96*128]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(tc.check(s.labels_idx == ones(7,1)));
            assert(s.samples_count == 7);
            assert(s.features_count == 96*128);
            assert(tc.check(size(s.images) == [96 128 1 7]));
            assert(tc.tensor(s.images,4) && tc.unitreal(s.images));
            assert(s.layers_count == 1);
            assert(s.row_count == 96);
            assert(s.col_count == 128);
            assert(all(all(s.samples == datasets.image.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('    With mode "color" and forced size.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small','color',[96 128]);
            
            assert(length(s.classes) == 1);
            assert(strcmp(s.classes{1},'none'));
            assert(s.classes_count == 1);
            assert(all(size(s.samples) == [7 3*96*128]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(tc.check(s.labels_idx == ones(7,1)));
            assert(s.samples_count == 7);
            assert(s.features_count == 3*96*128);
            assert(tc.check(size(s.images) == [96 128 3 7]));
            assert(tc.tensor(s.images,4) && tc.unitreal(s.images));
            assert(s.layers_count == 3);
            assert(s.row_count == 96);
            assert(s.col_count == 128);
            assert(all(all(s.samples == datasets.image.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('    With heterogenous directory.\n');
            
            s = datasets.image.load_from_dir('../data/test/scenes_small/heterogeneous_dir');
            
            assert(length(s.classes) == 1);
            assert(strcmp(s.classes{1},'none'));
            assert(s.classes_count == 1);
            assert(all(size(s.samples) == [2 192*256]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(tc.check(s.labels_idx == ones(2,1)));
            assert(s.samples_count == 2);
            assert(s.features_count == 192*256);
            assert(tc.check(size(s.images) == [192 256 1 2]));
            assert(tc.tensor(s.images,4) && tc.unitreal(s.images));
            assert(s.layers_count == 1);
            assert(s.row_count == 192);
            assert(s.col_count == 256);
            assert(all(all(s.samples == datasets.image.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('    With invalid external inputs.\n');
            
            try
                s = datasets.image.load_from_dir('../data/test/scenes_small_aaa');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('      Passes "No such file or directory!"\n');
                else
                    assert(false);
                end
            end
            
            try
                !chmod a-r ../data/test/scenes_small
                s = datasets.image.load_from_dir('../data/test/scenes_small');
                !chmod a+r ../data/test/scenes_small
                assert(false);
            catch exp
                !chmod a+r ../data/test/scenes_small
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('      Passes "Permission denied!"\n');
                else
                    assert(false);
                end
            end
            
            try
                s = datasets.image.load_from_dir('../data/test/scenes_small/empty_dir');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('      Passes "Empty directory!"\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except display;
            
            fprintf('  Function "load_mnist".\n');
            
            fprintf('    With MNIST test data.\n');
            
            s = datasets.image.load_mnist('../data/test/mnist/t10k-images-idx3-ubyte','../data/test/mnist/t10k-labels-idx1-ubyte');
            
            assert(length(s.classes) == 10);
            assert(strcmp(s.classes{1},'0'));
            assert(strcmp(s.classes{2},'1'));
            assert(strcmp(s.classes{3},'2'));
            assert(strcmp(s.classes{4},'3'));
            assert(strcmp(s.classes{5},'4'));
            assert(strcmp(s.classes{6},'5'));
            assert(strcmp(s.classes{7},'6'));
            assert(strcmp(s.classes{8},'7'));
            assert(strcmp(s.classes{9},'8'));
            assert(strcmp(s.classes{10},'9'));
            assert(s.classes_count == 10);
            assert(all(size(s.samples) == [10000 28*28]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(tc.check(size(s.labels_idx) == [10000 1]));
            assert(tc.vector(s.labels_idx) && tc.labels_idx(s.labels_idx,s.classes));
            assert(s.samples_count == 10000);
            assert(s.features_count == 28*28);
            assert(tc.check(size(s.images) == [28 28 1 10000]));
            assert(tc.tensor(s.images,4) && tc.unitreal(s.images));
            assert(s.layers_count == 1);
            assert(s.row_count == 28);
            assert(s.col_count == 28);
            assert(all(all(s.samples == datasets.image.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('    With invalid external inputs.\n');
            
            try
                s = datasets.image.load_mnist('../data/test/mnist/train-images-idx3-ubyte_aaa','../data/test/mnist/train-labels-idx1-ubyte');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not load images in "../data/test/mnist/train-images-idx3-ubyte_aaa": No such file or directory!')
                    fprintf('      Passes "No such file or directory!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s = datasets.image.load_mnist('../data/test/mnist/train-images-idx3-ubyte','../data/test/mnist/train-labels-idx1-ubyte_aaa');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not load labels in "../data/test/mnist/train-labels-idx1-ubyte_aaa": No such file or directory!')
                    fprintf('      Passes "No such file or directory!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                !chmod a-r ../data/test/mnist/train-images-idx3-ubyte
                s = datasets.image.load_mnist('../data/test/mnist/train-images-idx3-ubyte','../data/test/mnist/train-labels-idx1-ubyte');
                !chmod a+r ../data/test/mnist/train-images-idx3-ubyte
                assert(false);
            catch exp
                !chmod a+r ../data/test/mnist/train-images-idx3-ubyte
                if strcmp(exp.message,'Could not load images in "../data/test/mnist/train-images-idx3-ubyte": Permission denied!')
                    fprintf('      Passes "Permission denied!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                !chmod a-r ../data/test/mnist/train-labels-idx1-ubyte
                s = datasets.image.load_mnist('../data/test/mnist/train-images-idx3-ubyte','../data/test/mnist/train-labels-idx1-ubyte');
                !chmod a+r ../data/test/mnist/train-labels-idx1-ubyte
                assert(false);
            catch exp
                !chmod a+r ../data/test/mnist/train-labels-idx1-ubyte
                if strcmp(exp.message,'Could not load labels in "../data/test/mnist/train-labels-idx1-ubyte": Permission denied!')
                    fprintf('      Passes "Permission denied!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s = datasets.image.load_mnist('../data/test/scenes_small/scenes_small1.jpg','../data/test/mnist/train-labels-idx1-ubyte');
            catch exp
                if strcmp(exp.message,'Images file "../data/test/scenes_small/scenes_small1.jpg" not in MNIST format!')
                    fprintf('      Passes "Not in MNIST format!" for images file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s = datasets.image.load_mnist('../data/test/mnist/train-images-idx3-ubyte','../data/test/scenes_small/scenes_small1.jpg');
            catch exp
                if strcmp(exp.message,'Labels file "../data/test/scenes_small/scenes_small1.jpg" not in MNIST format!')
                    fprintf('      Passes "Not in MNIST format!" for labels file.\n');
                else
                    assert(false);
                end
            end
            
            try
                s = datasets.image.load_mnist('../data/test/mnist/train-images-idx3-ubyte','../data/test/mnist/t10k-labels-idx1-ubyte');
            catch exp
                if strcmp(exp.message,'Different number of labels in "../data/test/mnist/t10k-labels-idx1-ubyte" for images in "../data/test/mnist/train-images-idx3-ubyte"!')
                    fprintf('      Passes "Different number of labels!".\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except display;
        end
    end
end