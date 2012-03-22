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
    end
    
    methods (Static,Access=public)
        function [new_image] = do_subsamples(obj,index)
            new_image = datasets.image(obj.classes,obj.images(:,:,:,index),obj.labels_idx(index));
        end
        
        function [new_image] = from_data(images,labels)
            assert(tc.tensor(images,4) && tc.unitreal(images));
            assert(tc.vector(labels) && tc.match_dims(images,labels,4) && tc.labels(labels));
            
            [labels_idx_t,classes_t] = grp2idx(labels);
            new_image = datasets.image(classes_t,images,labels_idx_t);
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
    end
    
    methods (Static,Access=public)
        function test(~)
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
            
            clearvars -except display
            
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
        end
    end
end