classdef dataset
    properties (GetAccess=public,SetAccess=immutable)
        classes;
        classes_count;
        samples;
        labels_idx;
        samples_count;
        features_count;
    end
    
    methods (Access=public)
        function [obj] = dataset(classes,samples,labels_idx)
            assert(tc.vector(classes) && tc.labels(classes));
            assert(tc.matrix(samples) && tc.number(samples));
            assert(tc.vector(labels_idx) && tc.match_dims(samples,labels_idx,1) && ...
                   tc.labels_idx(labels_idx,classes));

            obj.classes = utils.force_col(classes);
            obj.classes_count = length(classes);
            obj.samples = samples;
            obj.labels_idx = utils.force_col(labels_idx);
            obj.samples_count = size(samples,1);
            obj.features_count = size(samples,2);
        end
        
        function [o] = eq(obj,another_dataset)
            assert(tc.scalar(obj) && tc.dataset(obj));
            assert(tc.scalar(another_dataset) && tc.dataset(another_dataset));
            
            o = true;
            o = o && utils.same_classes(obj.classes,another_dataset.classes);
            o = o && obj.compatible(another_dataset);
            o = o && tc.check(size(obj.samples) == size(another_dataset.samples));
            o = o && tc.check(obj.samples == another_dataset.samples);
            o = o && tc.check(size(obj.labels_idx) == size(another_dataset.labels_idx));
            o = o && tc.check(obj.labels_idx == another_dataset.labels_idx);
            o = o && (obj.samples_count == another_dataset.samples_count);
            o = o && (obj.features_count == another_dataset.features_count);
        end
        
        function [o] = ne(obj,another_dataset)
            assert(tc.scalar(obj) && tc.dataset(obj));
            assert(tc.scalar(another_dataset) && tc.dataset(another_dataset));
            
            o = ~obj.eq(another_dataset);
        end
        
        function [o] = compatible(obj,another_dataset)
            assert(tc.scalar(obj) && tc.dataset(another_dataset));
            assert(tc.scalar(another_dataset) && tc.dataset(another_dataset));
            
            o = true;
            o = o && utils.same_classes(obj.classes,another_dataset.classes);
            o = o && (obj.features_count == another_dataset.features_count);
        end

        function [tr_index,ts_index] = partition(obj,type,param)
            assert(tc.scalar(obj) && tc.dataset(obj));
            assert(tc.scalar(type) && tc.string(type) && (strcmp(type,'kfold') || strcmp(type,'holdout')));
            assert(tc.scalar(param) && tc.number(param) && ...
                   ((strcmp(type,'kfold') && tc.natural(param) && (param >= 2)) || ...
                    (strcmp(type,'holdout') && tc.unitreal(param))));
            
            partition = cvpartition(obj.labels_idx,type,param);
            
            tr_index = false(obj.samples_count,partition.NumTestSets);
            ts_index = false(obj.samples_count,partition.NumTestSets);
            
            for i = 1:partition.NumTestSets
                tr_index(:,i) = training(partition,i)';
                ts_index(:,i) = test(partition,i)';
            end
        end
        
        function [new_dataset] = subsamples(obj,index)
            assert(tc.scalar(obj) && tc.dataset(obj));
            assert(tc.vector(index) && ...
                   ((tc.logical(index) && tc.match_dims(obj.samples,index,1)) || ...
                    (tc.natural(index) && tc.check(index > 0 & index <= obj.samples_count))));
            
            new_dataset = dataset(obj.classes,obj.samples(index,:),obj.labels_idx(index));
        end
    end
    
    methods (Static,Access=public)
        function [new_dataset] = from_data(samples,labels)
            assert(tc.matrix(samples) && tc.number(samples));
            assert(tc.vector(labels) && tc.match_dims(samples,labels,1) && tc.labels(labels));
            
            [labels_idx_t,classes_t] = grp2idx(labels);
            new_dataset = dataset(classes_t,samples,labels_idx_t);
        end
        
        function [new_dataset] = from_fulldata(classes,samples,labels_idx)
            assert(tc.vector(classes) && tc.labels(classes));
            assert(tc.matrix(samples) && tc.number(samples));
            assert(tc.vector(labels_idx) && tc.match_dims(samples,labels_idx,1) && ...
                   tc.labels_idx(labels_idx,classes));
               
            new_dataset = dataset(classes,samples,labels_idx);
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "dataset".\n');
            
            fprintf('  Proper construction.\n');
            
            A = [1 2 3 4;
                 1 2 4 3;
                 1 3 2 4;
                 1 3 4 2;
                 1 4 2 3;
                 1 4 3 2;
                 2 1 3 4;
                 2 1 4 3;
                 2 3 1 4;
                 2 3 4 1;
                 2 4 1 3;
                 2 4 3 1];             
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
             
            s = dataset({'1' '2' '3'},A,c);
            
            assert(length(s.classes) == 3);
            assert(strcmp(s.classes{1},'1'));
            assert(strcmp(s.classes{2},'2'));
            assert(strcmp(s.classes{3},'3'));
            assert(s.classes_count == 3);
            assert(tc.check(s.samples == A));
            assert(tc.check(s.labels_idx == c'));
            assert(s.samples_count == 12);
            assert(s.features_count == 4);
            
            clearvars -except display;
            
            fprintf('  Functions "eq" and "ne".\n');
            
            s1 = dataset({'1' '2'},[1 2 3; 1 3 2],[1 2]);
            s2 = dataset({'1' '2'},[1 2 3; 1 3 2],[1 2]);
            s3 = dataset({'1' '2' '3'},[1 2 3; 1 3 2],[1 2]);
            s4 = dataset({'hello' 'world'},[1 2 3; 1 3 2],[1 2]);
            s5 = dataset([1 2],[1 2 3; 1 3 2],[1 2]);
            s6 = dataset([true false],[1 2 3; 1 3 2],[1 2]);
            s7 = dataset({'1' '2'},[1 2 3 4; 1 3 2 4],[1 2]);
            s8 = dataset({'1' '2'},[1 2 3; 1 3 2; 2 1 3],[1 2 2]);
            s9 = dataset({'1' '2'},[1 2 3; 1 3 3],[1 2]);
            s10 = dataset({'1' '2'},[1 2 3; 1 3 2],[1 1]);
            
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
            
            s1 = dataset({'1' '2'},rand(100,10),randi(2,100,1));
            s2 = dataset({'1' '2'},rand(150,10),randi(2,150,1));
            s3 = dataset({'1' '2' '3'},rand(50,10),randi(2,50,1));
            s4 = dataset({'hello' 'world'},rand(50,10),randi(2,50,1));
            s5 = dataset([1 2],rand(50,10),randi(2,50,1));
            s6 = dataset([true false],rand(50,10),randi(2,50,1));
            s7 = dataset({'1' '2'},rand(50,15),randi(2,50,1));
            
            assert(s1.compatible(s2) == true);
            assert(s1.compatible(s3) == false);
            assert(s1.compatible(s4) == false);
            assert(s1.compatible(s5) == false);
            assert(s1.compatible(s6) == false);
            assert(s1.compatible(s7) == false);
            
            clearvars -except display;
            
            fprintf('  Functions "partition" and "subsamples".\n');
            
            fprintf('    2-fold partition and call to "subsamples" with boolean indices.\n');
            
            A = [1 2 3 4;
                 1 2 4 3;
                 1 3 2 4;
                 1 3 4 2;
                 1 4 2 3;
                 1 4 3 2;
                 2 1 3 4;
                 2 1 4 3;
                 2 3 1 4;
                 2 3 4 1;
                 2 4 1 3;
                 2 4 3 1];             
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
             
            s = dataset({'1' '2' '3'},A,c);
            
            [tr_f,ts_f] = s.partition('kfold',2);
            
            s_f11 = s.subsamples(tr_f(:,1));
            
            assert(length(s_f11.classes) == 3);
            assert(strcmp(s_f11.classes{1},'1'));
            assert(strcmp(s_f11.classes{2},'2'));
            assert(strcmp(s_f11.classes{3},'3'));
            assert(s_f11.classes_count == 3);
            assert(tc.check(s_f11.samples == A(tr_f(:,1),:)));
            assert(tc.check(s_f11.labels_idx == c(tr_f(:,1))'));
            assert(s_f11.samples_count == 6);
            assert(s_f11.features_count == 4);
            
            s_f12 = s.subsamples(ts_f(:,1));
            
            assert(length(s_f12.classes) == 3);
            assert(strcmp(s_f12.classes{1},'1'));
            assert(strcmp(s_f12.classes{2},'2'));
            assert(strcmp(s_f12.classes{3},'3'));
            assert(s_f12.classes_count == 3);
            assert(tc.check(s_f12.samples == A(ts_f(:,1),:)));
            assert(tc.check(s_f12.labels_idx == c(ts_f(:,1))'));
            assert(s_f12.samples_count == 6);
            assert(s_f12.features_count == 4);
            
            s_f21 = s.subsamples(tr_f(:,2));

            assert(length(s_f21.classes) == 3);
            assert(strcmp(s_f21.classes{1},'1'));
            assert(strcmp(s_f21.classes{2},'2'));
            assert(strcmp(s_f21.classes{3},'3'));
            assert(s_f21.classes_count == 3);
            assert(tc.check(s_f21.samples == A(tr_f(:,2),:)));
            assert(tc.check(s_f21.labels_idx == c(tr_f(:,2))'));
            assert(s_f21.samples_count == 6);
            assert(s_f21.features_count == 4);
            
            s_f22 = s.subsamples(ts_f(:,2));
            
            assert(length(s_f22.classes) == 3);
            assert(strcmp(s_f22.classes{1},'1'));
            assert(strcmp(s_f22.classes{2},'2'));
            assert(strcmp(s_f22.classes{3},'3'));
            assert(s_f22.classes_count == 3);
            assert(tc.check(s_f22.samples == A(ts_f(:,2),:)));
            assert(tc.check(s_f22.labels_idx == c(ts_f(:,2))'));
            assert(s_f22.samples_count == 6);
            assert(s_f22.features_count == 4);
            
            clearvars -except display;
            
            fprintf('    Holdout partition with p=0.33 and call to "subsamples" with boolean indices.\n');
            
            A = [1 2 3 4;
                 1 2 4 3;
                 1 3 2 4;
                 1 3 4 2;
                 1 4 2 3;
                 1 4 3 2;
                 2 1 3 4;
                 2 1 4 3;
                 2 3 1 4;
                 2 3 4 1;
                 2 4 1 3;
                 2 4 3 1];             
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
             
            s = dataset({'1' '2' '3'},A,c);
            
            [tr_h,ts_h] = s.partition('holdout',0.33);
            
            s_h1 = s.subsamples(tr_h);
            
            assert(length(s_h1.classes) == 3);
            assert(strcmp(s_h1.classes{1},'1'));
            assert(strcmp(s_h1.classes{2},'2'));
            assert(strcmp(s_h1.classes{2},'2'));
            assert(s_h1.classes_count == 3);
            assert(tc.check(s_h1.samples == A(tr_h,:)));
            assert(tc.check(s_h1.labels_idx == c(tr_h)'));
            assert(s_h1.samples_count == 9);
            assert(s_h1.features_count == 4);
            
            s_h2 = s.subsamples(ts_h);
            
            assert(length(s_h2.classes) == 3);
            assert(strcmp(s_h2.classes{1},'1'));
            assert(strcmp(s_h2.classes{2},'2'));
            assert(strcmp(s_h2.classes{2},'2'));
            assert(s_h2.classes_count == 3);
            assert(tc.check(s_h2.samples == A(ts_h,:)));
            assert(tc.check(s_h2.labels_idx == c(ts_h)'));
            assert(s_h2.samples_count == 3);
            assert(s_h2.features_count == 4);
            
            clearvars -except display;
            
            fprintf('    Call to "subsamples" with natural indices.\n');
            
            A = [1 2 3 4;
                 1 2 4 3;
                 1 3 2 4;
                 1 3 4 2;
                 1 4 2 3;
                 1 4 3 2;
                 2 1 3 4;
                 2 1 4 3;
                 2 3 1 4;
                 2 3 4 1;
                 2 4 1 3;
                 2 4 3 1];             
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
             
            s = dataset({'1' '2' '3'},A,c);
            
            s_fi = s.subsamples(1:2:12);
            
            assert(length(s_fi.classes) == 3);
            assert(strcmp(s_fi.classes{1},'1'));
            assert(strcmp(s_fi.classes{2},'2'));
            assert(strcmp(s_fi.classes{3},'3'));
            assert(s_fi.classes_count == 3);
            assert(tc.check(s_fi.samples == A(1:2:12,:)));
            assert(tc.check(s_fi.labels_idx == c(1:2:12)'));
            assert(s_fi.samples_count == 6);
            assert(s_fi.features_count == 4);
            
            clearvars -except display;
            
            fprintf('    Call to "subsamples" with natural indices and redundant selection.\n');
            
            A = [1 2 3 4;
                 1 2 4 3;
                 1 3 2 4;
                 1 3 4 2;
                 1 4 2 3;
                 1 4 3 2;
                 2 1 3 4;
                 2 1 4 3;
                 2 3 1 4;
                 2 3 4 1;
                 2 4 1 3;
                 2 4 3 1];             
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
             
            s = dataset({'1' '2' '3'},A,c);
            
            s_fo = s.subsamples([1:12,1:12]);
            
            assert(length(s_fo.classes) == 3);
            assert(strcmp(s_fo.classes{1},'1'));
            assert(strcmp(s_fo.classes{2},'2'));
            assert(strcmp(s_fo.classes{3},'3'));
            assert(s_fo.classes_count == 3);
            assert(tc.check(s_fo.samples == [A;A]));
            assert(tc.check(s_fo.labels_idx == [c c]'));
            assert(s_fo.samples_count == 24);
            assert(s_fo.features_count == 4);
            
            clearvars -except display;
            
            fprintf('  Function "from_data".\n');
            
            A = [1 2 3 4;
                 1 2 4 3;
                 1 3 2 4;
                 1 3 4 2;
                 1 4 2 3;
                 1 4 3 2;
                 2 1 3 4;
                 2 1 4 3;
                 2 3 1 4;
                 2 3 4 1;
                 2 4 1 3;
                 2 4 3 1];             
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
            
            s = dataset.from_data(A,c);
            
            assert(length(s.classes) == 3);
            assert(strcmp(s.classes{1},'1'));
            assert(strcmp(s.classes{2},'2'));
            assert(strcmp(s.classes{3},'3'));
            assert(s.classes_count == 3);
            assert(tc.check(s.samples == A));
            assert(tc.check(s.labels_idx == c'));
            assert(s.samples_count == 12);
            assert(s.features_count == 4);
            
            clearvars -except display;
            
            fprintf('  Function "from_fulldata".\n');
            
            A = [1 2 3 4;
                 1 2 4 3;
                 1 3 2 4;
                 1 3 4 2;
                 1 4 2 3;
                 1 4 3 2;
                 2 1 3 4;
                 2 1 4 3;
                 2 3 1 4;
                 2 3 4 1;
                 2 4 1 3;
                 2 4 3 1];             
            c = [1 2 3 1 2 3 1 2 3 1 2 3];
             
            s = dataset.from_fulldata({'1' '2' '3'},A,c);
            
            assert(tc.dataset(s));
            assert(length(s.classes) == 3);
            assert(strcmp(s.classes{1},'1'));
            assert(strcmp(s.classes{2},'2'));
            assert(strcmp(s.classes{3},'3'));
            assert(s.classes_count == 3);
            assert(tc.check(s.samples == A));
            assert(tc.check(s.labels_idx == c'));
            assert(s.samples_count == 12);
            assert(s.features_count == 4);
            
            clearvars -except display;
        end
    end
end
