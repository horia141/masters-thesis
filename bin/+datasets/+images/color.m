classdef color < datasets.image
    methods (Access=public)
        function [obj] = color(classes,images,labels_idx)
            assert(tc.vector(classes) && tc.labels(classes));
            assert(tc.tensor(images,4) && (size(images,3) == 3) && tc.unitreal(images));
            assert(tc.vector(labels_idx) && tc.match_dims(images,labels_idx,4) && ...
                   tc.labels_idx(labels_idx,classes));
               
            obj = obj@datasets.image(classes,images,labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function [new_color] = do_subsamples(obj,index)
            new_color = datasets.images.color(obj.classes,obj.images(:,:,:,index),obj.labels_idx(index));
        end
        
        function [new_color] = from_data(images,labels)
            assert(tc.tensor(images,4) && (size(images,3) == 3) && tc.unitreal(images));
            assert(tc.vector(labels) && tc.match_dims(images,labels,4) && tc.labels(labels));
            
            [labels_idx_t,classes_t] = grp2idx(labels);
            new_color = datasets.images.color(classes_t,images,labels_idx_t);
        end

        function [new_color] = load_from_dir(images_dir_path,force_size)
            assert(tc.scalar(images_dir_path) && tc.string(images_dir_path));
            assert(~exist('force_size','var') || ...
                   (tc.vector(force_size) && (length(force_size) == 2) && ...
                    tc.natural(force_size) && tc.check(force_size > 1)));
               
            paths = dir(images_dir_path);
            images_t = [];
            current_image = 1;
            
            for i = 1:length(paths)
                try
                    image = imread(fullfile(images_dir_path,paths(i).name));
                    image = double(image) ./ 255;
                    
                    if size(image,3) ~= 3
                        continue;
                    end
                    
                    if exist('force_size','var')
                        image = imresize(image,force_size);
                        
                        % Correct small domain overflows caused by resizing.
                        image = utils.clamp_images_to_unit(image);
                    end
                    
                    if (current_image > 1) && ...
                        (~all(size(image) == size(images_t(:,:,:,1))))
                        throw(MException('master:datasets:images:color:load_from_dir:NoLoad',...
                                         'Images are of different sizes!'));
                    end

                    images_t(:,:,:,current_image) = image;
                    current_image = current_image + 1;
                catch exp
                    if isempty(regexp(exp.identifier,'MATLAB:imread:.*','ONCE'))
                        throw(MException('master:datasets:images:color:load_from_dir:NoLoad',exp.message));
                    end
                end
            end
            
            if isempty(images_t)
                throw(MException('master:datasets:images:color:load_from_dir:NoLoad',...
                                 'Could not find any acceptable images in the directory.'));
            end
            
            new_color = datasets.images.color({'none'},images_t,ones(current_image - 1,1));
        end
    end
    
    methods (Static,Access=public)        
        function test(display)
            fprintf('Testing "datasets.images.color".\n');
            
            fprintf('  Proper construction.\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
                        
            s = datasets.images.color({'1' '2' '3'},A,c);
            
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
            assert(all(all(s.samples == datasets.images.color.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('  Functions "eq" and "ne".\n');
            
            s1 = datasets.images.color({'1' '2'},cat(3,cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                       cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                       cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s2 = datasets.images.color({'1' '2'},cat(3,cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                       cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                       cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s3 = datasets.images.color({'1' '2' '3'},cat(3,cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                           cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                           cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s4 = datasets.images.color({'true' 'false'},cat(3,cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                              cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                              cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s5 = datasets.images.color([1 2],cat(3,cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                   cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                   cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s6 = datasets.images.color([true false],cat(3,cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                          cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                          cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 2]);
            s7 = datasets.images.color({'1' '2'},cat(3,cat(4,[0.1 0.2 0.3; 0.1 0.3 0.2],[0.1 0.2 0.4; 0.1 0.4 0.2]),...
                                                       cat(4,[0.1 0.2 0.3; 0.1 0.3 0.2],[0.1 0.2 0.4; 0.1 0.4 0.2]),...
                                                       cat(4,[0.1 0.2 0.3; 0.1 0.3 0.2],[0.1 0.2 0.4; 0.1 0.4 0.2])),[1 2]);
            s8 = datasets.images.color({'1' '2'},cat(3,cat(4,[0.1 0.2 0.2 0.1],[0.1 0.3 0.3 0.1]),...
                                                       cat(4,[0.1 0.2 0.2 0.1],[0.1 0.3 0.3 0.1]),...
                                                       cat(4,[0.1 0.2 0.2 0.1],[0.1 0.3 0.3 0.1])),[1 2]);
            s9 = datasets.images.color({'1' '2'},cat(3,cat(4,[0.1 0.2; 0.2 0.2],[0.1 0.3; 0.3 0.1]),...
                                                       cat(4,[0.1 0.2; 0.2 0.2],[0.1 0.3; 0.3 0.1]),...
                                                       cat(4,[0.1 0.2; 0.2 0.2],[0.1 0.3; 0.3 0.1])),[1 2]);
            s10 = datasets.images.color({'1' '2'},cat(3,cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                        cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1]),...
                                                        cat(4,[0.1 0.2; 0.2 0.1],[0.1 0.3; 0.3 0.1])),[1 1]);
            
            assert(s1 == s2);
            assert(s1 ~= s3);
            assert(s1 ~= s4);
            assert(s1 ~= s5);
            assert(s1 ~= s6);
            assert(s1 ~= s7);
            assert(s1 ~= s8);
            assert(s1 ~= s9);
            assert(s1 ~= s10);
            
            clearvars -except display
            
            fprintf('  Function "compatible".\n');
            
            s1 = datasets.images.color({'1' '2'},rand(10,10,3,3),randi(2,3,1));
            s2 = datasets.images.color({'1' '2'},rand(10,10,3,4),randi(2,4,1));
            s3 = datasets.images.color({'1' '2' '3'},rand(10,10,3,3),randi(2,3,1));
            s4 = datasets.images.color({'hello' 'world'},rand(10,10,3,3),randi(2,3,1));
            s5 = datasets.images.color([1 2],rand(10,10,3,3),randi(2,3,1));
            s6 = datasets.images.color([true false],rand(10,10,3,3),randi(2,3,1));
            s7 = datasets.images.color({'1' '2'},rand(12,10,3,3),randi(2,3,1));
            s8 = datasets.images.color({'1' '2'},rand(10,12,3,3),randi(2,3,1));
            s9 = datasets.images.color({'1' '2'},rand(4,25,3,4),randi(2,4,1));
            
            assert(s1.compatible(s2) == true);
            assert(s1.compatible(s3) == false);
            assert(s1.compatible(s4) == false);
            assert(s1.compatible(s5) == false);
            assert(s1.compatible(s6) == false);
            assert(s1.compatible(s7) == false);
            assert(s1.compatible(s8) == false);
            assert(s1.compatible(s9) == false);
            
            clearvars -except display;            
            
            fprintf('  Functions "partition" and "subsamples".\n');
            
            fprintf('    2-fold partition and call to "subsamples" with boolean indices.\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
            
            s = datasets.images.color({'1' '2' '3'},A,c);
            
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
            assert(all(all(s_f11.samples == datasets.images.color.to_samples(s_f11.images))));
            
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
            assert(all(all(s_f12.samples == datasets.images.color.to_samples(s_f12.images))));
            
            s_f21 = s.subsamples(tr_f(:,2));
            
            assert(length(s_f21.classes) == 3);
            assert(strcmp(s_f21.classes{1},'1'));
            assert(strcmp(s_f21.classes{2},'2'));
            assert(strcmp(s_f21.classes{3},'3'));
            assert(s_f21.classes_count == 3);
            assert(tc.check(s_f21.samples == A_s(tr_f(:,2),:)));
            assert(tc.check(s_f21.labels_idx == c(tr_f(:,2))'));
            assert(s_f21.samples_count == 6);
            assert(s_f21.features_count == 300);
            assert(tc.check(s_f21.images == A(:,:,:,tr_f(:,2))));
            assert(s_f21.layers_count == 3);
            assert(s_f21.row_count == 10);
            assert(s_f21.col_count == 10);
            assert(all(all(s_f21.samples == datasets.images.color.to_samples(s_f21.images))));
            
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
            assert(all(all(s_f22.samples == datasets.images.color.to_samples(s_f22.images))));
            
            clearvars -except display;
            
            fprintf('    Holdout partition with p=0.33 and call to "subsamples" with boolean indices.\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
            
            s = datasets.images.color({'1' '2' '3'},A,c);
            
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
            assert(all(all(s_h1.samples == datasets.images.color.to_samples(s_h1.images))));
            
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
            assert(s_h1.layers_count == 3);
            assert(s_h2.row_count == 10);
            assert(s_h2.col_count == 10);
            assert(all(all(s_h2.samples == datasets.images.color.to_samples(s_h2.images))));
            
            clearvars -except display;
            
            fprintf('    Call to "subsamples" with natural indices.\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
            
            s = datasets.images.color({'1' '2' '3'},A,c);
            
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
            assert(all(all(s_fi.samples == datasets.images.color.to_samples(s_fi.images))));
            
            clearvars -except display;
            
            fprintf('    Call to "subsamples" with natural indices and redundant selection.\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
            
            s = datasets.images.color({'1' '2' '3'},A,c);
            
            s_fo = s.subsamples([1:12,1:12]);
            
            assert(length(s_fo.classes) == 3);
            assert(strcmp(s_fo.classes{1},'1'));
            assert(strcmp(s_fo.classes{2},'2'));
            assert(strcmp(s_fo.classes{3},'3'));
            assert(s_fo.classes_count == 3);
            assert(tc.check(s_fo.samples == [A_s;A_s]));
            assert(all(s_fo.labels_idx == [c c]'));
            assert(s_fo.samples_count == 24);
            assert(s_fo.features_count == 300);
            assert(tc.check(s_fo.images == cat(4,A,A)));
            assert(s_fo.layers_count == 3);
            assert(s_fo.row_count == 10);
            assert(s_fo.col_count == 10);
            assert(all(all(s_fo.samples == datasets.images.color.to_samples(s_fo.images))));
            
            clearvars -except display;
            
            fprintf('  Function "from_data".\n');
            
            A = rand(10,10,3,12);
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            A_s = zeros(12,300);
            for i = 1:12
                A_s(i,:) = reshape(A(:,:,:,i),[1 300]);
            end
                        
            s = datasets.images.color.from_data(A,c);
            
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
            assert(all(all(s.samples == datasets.images.color.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('  Function "load_from_dir".\n');
            
            fprintf('    With test data.\n');
            
            s = datasets.images.color.load_from_dir('../data/test/scenes_small');
            
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
            assert(all(all(s.samples == datasets.images.color.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('    With forced size on test data.\n');
            
            s = datasets.images.color.load_from_dir('../data/test/scenes_small',[96 128]);
            
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
            assert(all(all(s.samples == datasets.images.color.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('    With heterogenous directory.\n');
            
            s = datasets.images.color.load_from_dir('../data/test/scenes_small/heterogeneous_dir');
            
            assert(length(s.classes) == 1);
            assert(strcmp(s.classes{1},'none'));
            assert(s.classes_count == 1);
            assert(all(size(s.samples) == [2 3*192*256]));
            assert(tc.matrix(s.samples) && tc.unitreal(s.samples));
            assert(tc.check(s.labels_idx == ones(2,1)));
            assert(s.samples_count == 2);
            assert(s.features_count == 3*192*256);
            assert(tc.check(size(s.images) == [192 256 3 2]));
            assert(tc.tensor(s.images,4) && tc.unitreal(s.images));
            assert(s.layers_count == 3);
            assert(s.row_count == 192);
            assert(s.col_count == 256);
            assert(all(all(s.samples == datasets.images.color.to_samples(s.images))));
            
            clearvars -except display;
            
            fprintf('    With invalid external inputs.\n');
            
            try
                s = datasets.images.color.load_from_dir('../data/test/scenes_small_aaa');
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
                s = datasets.images.color.load_from_dir('../data/test/scenes_small');
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
                s = datasets.images.color.load_from_dir('../data/test/scenes_small/empty_dir');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not find any acceptable images in the directory.')
                    fprintf('      Passes "Empty directory!"\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except display;
        end
    end
end
