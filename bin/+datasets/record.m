classdef record < dataset
    methods (Access=public)
        function [obj] = record(classes,samples,labels_idx)
            assert(tc.vector(classes) && tc.labels(classes));
            assert(tc.matrix(samples) && tc.number(samples));
            assert(tc.vector(labels_idx) && tc.match_dims(samples,labels_idx,1) && ...
                   tc.labels_idx(labels_idx,classes));
               
            obj = obj@dataset(classes,samples,labels_idx);
        end
        
        function [o] = eq(obj,another_record)
            assert(tc.scalar(obj) && tc.datasets_record(obj));
            assert(tc.scalar(another_record) && tc.datasets_record(another_record));
            
            o = true;
            o = o && obj.eq@dataset(another_record);
        end
        
        function [o] = compatible(obj,another_record)
            assert(tc.scalar(obj) && tc.datasets_record(another_record));
            assert(tc.scalar(another_record) && tc.datasets_record(another_record));
            
            o = true;
            o = o && obj.compatible@dataset(another_record);
        end
        
        function [new_record] = subsamples(obj,index)
            assert(tc.scalar(obj) && tc.datasets_record(obj));
            assert(tc.vector(index) && ...
                   ((tc.logical(index) && tc.match_dims(obj.samples,index,1)) || ...
                    (tc.natural(index) && tc.check(index > 0 & index <= obj.samples_count))));
            
            new_record = datasets.record(obj.classes,obj.samples(index,:),obj.labels_idx(index));
        end
    end
    
    methods (Static,Access=public)
        function [new_record] = from_data(samples,labels)
            assert(tc.matrix(samples) && tc.number(samples));
            assert(tc.vector(labels) && tc.match_dims(samples,labels,1) && tc.labels(labels));
            
            [labels_idx_t,classes_t] = grp2idx(labels);
            new_record = datasets.record(classes_t,samples,labels_idx_t);
        end
        
        function [new_record] = from_fulldata(classes,samples,labels_idx)
            assert(tc.vector(classes) && tc.labels(classes));
            assert(tc.matrix(samples) && tc.number(samples));
            assert(tc.vector(labels_idx) && tc.match_dims(samples,labels_idx,1) && ...
                   tc.labels_idx(labels_idx,classes));
               
            new_record = datasets.record(classes,samples,labels_idx);
        end
        
        function [new_record] = load_csvfile(csvfile_path,labels_format,data_format,delimiter)
            assert(tc.scalar(csvfile_path) && tc.string(csvfile_path));
            assert(tc.scalar(labels_format) && tc.string(labels_format));
            assert(tc.scalar(data_format) && tc.string(data_format));
            assert(~exist('delimiter','var') || (tc.scalar(delimiter) && tc.string(delimiter)));
            
            if exist('delimiter','var')
                delimiter_t = delimiter;
            else
                delimiter_t = ',';
            end
            
            try
                [csvfile_fid,csvfile_msg] = fopen(csvfile_path,'rt');
                
                if csvfile_fid == -1
                    throw(MException('master:datasets:record:load_csvfile:NoLoad',...
                             sprintf('Could not load csv file "%s": %s!',csvfile_path,csvfile_msg)))
                end
                
                samples_raw = textscan(csvfile_fid,strcat(labels_format,data_format),'delimiter',delimiter_t);
                fclose(csvfile_fid);
            catch exp
                throw(MException('master:datasets:record:load_csvfile:NoLoad',exp.message));
            end
            
            if ~tc.check(cellfun(@tc.number,samples_raw(:,2:end)))
                throw(MException('master:datasets:record:load_csvfile:InvalidFormat',...
                         sprintf('File "%s" has an invalid format!',csvfile_path)));
            end
            
            samples_t = cell2mat(samples_raw(:,2:end));
            labels_t = samples_raw{:,1};
            
            new_record = datasets.record.from_data(samples_t,labels_t);
        end
    end
    
    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "datasets.record".\n');
            
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
             
            s = datasets.record({'1' '2' '3'},A,c);
            
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
            
            s1 = datasets.record({'1' '2'},[1 2 3; 1 3 2],[1 2]);
            s2 = datasets.record({'1' '2'},[1 2 3; 1 3 2],[1 2]);
            s3 = datasets.record({'1' '2' '3'},[1 2 3; 1 3 2],[1 2]);
            s4 = datasets.record({'hello' 'world'},[1 2 3; 1 3 2],[1 2]);
            s5 = datasets.record([1 2],[1 2 3; 1 3 2],[1 2]);
            s6 = datasets.record([true false],[1 2 3; 1 3 2],[1 2]);
            s7 = datasets.record({'1' '2'},[1 2 3 4; 1 3 2 4],[1 2]);
            s8 = datasets.record({'1' '2'},[1 2 3; 1 3 2; 2 1 3],[1 2 2]);
            s9 = datasets.record({'1' '2'},[1 2 3; 1 3 3],[1 2]);
            s10 = datasets.record({'1' '2'},[1 2 3; 1 3 2],[1 1]);
            
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
            
            s1 = datasets.record({'1' '2'},rand(100,10),randi(2,100,1));
            s2 = datasets.record({'1' '2'},rand(150,10),randi(2,150,1));
            s3 = datasets.record({'1' '2' '3'},rand(50,10),randi(2,50,1));
            s4 = datasets.record({'hello' 'world'},rand(50,10),randi(2,50,1));
            s5 = datasets.record([1 2],rand(50,10),randi(2,50,1));
            s6 = datasets.record([true false],rand(50,10),randi(2,50,1));
            s7 = datasets.record({'1' '2'},rand(50,15),randi(2,50,1));
            
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
             
            s = datasets.record({'1' '2' '3'},A,c);
            
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
             
            s = datasets.record({'1' '2' '3'},A,c);
            
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
             
            s = datasets.record({'1' '2' '3'},A,c);
            
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
             
            s = datasets.record({'1' '2' '3'},A,c);
            
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
            
            s = datasets.record.from_data(A,c);
            
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
             
            s = datasets.record.from_fulldata({'1' '2' '3'},A,c);
            
            assert(tc.datasets_record(s));
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
            
            fprintf('  Function "load_csvfile".\n');
            
            fprintf('    With iris data.\n');
            
            s = datasets.record.load_csvfile('../data/test/iris/iris.csv','%s','%f%f%f%f',',');
            
            assert(length(s.classes) == 3);
            assert(strcmp(s.classes{1},'Iris-setosa'));
            assert(strcmp(s.classes{2},'Iris-versicolor'));
            assert(strcmp(s.classes{3},'Iris-virginica'));
            assert(s.classes_count == 3);
            assert(all(size(s.samples) == [150 4]));
            assert(tc.matrix(s.samples) && tc.number(s.samples));
            assert(tc.check(s.labels_idx == [1*ones(50,1);2*ones(50,1);3*ones(50,1)]));
            assert(s.samples_count == 150);
            assert(s.features_count == 4);
            
            clearvars -except display;
            
            fprintf('    With Wine data.\n');
            
            s = datasets.record.load_csvfile('../data/test/wine/wine.csv','%d','%f%f%f%f%f%f%f%f%f%f%f%f%f');
            
            assert(length(s.classes) == 3);
            assert(strcmp(s.classes{1},'1'));
            assert(strcmp(s.classes{2},'2'));
            assert(strcmp(s.classes{3},'3'));
            assert(s.classes_count == 3);
            assert(all(size(s.samples) == [178 13]));
            assert(tc.matrix(s.samples) && tc.number(s.samples));
            assert(tc.check(s.labels_idx == [1*ones(59,1);2*ones(71,1);3*ones(48,1)]));
            assert(s.samples_count == 178);
            assert(s.features_count == 13);
            
            clearvars -except display;
            
            fprintf('    With invalid external inputs.\n');
            
            try
                s = datasets.record.load_csvfile('../data/test/wine/wine_aaa.csv','%d','%f%f%f%f%f%f%f%f%f%f%f%f%f');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not load csv file "../data/test/wine/wine_aaa.csv": No such file or directory!')
                    fprintf('      Passes "No such file or directory!" test.\n');
                else
                    assert(false);
                end
            end
            
            try
                !chmod a-r ../data/test/wine/wine.csv
                s = datasets.record.load_csvfile('../data/test/wine/wine.csv','%d','%f%f%f%f%f%f%f%f%f%f%f%f%f');
                !chmod a+r ../data/test/wine/wine.csv
                assert(false);
            catch exp
                !chmod a+r ../data/test/wine/wine.csv
                if strcmp(exp.message,'Could not load csv file "../data/test/wine/wine.csv": Permission denied!')
                    fprintf('      Passes "Permission denied!" test.\n');
                else
                    assert(false);
                end
            end
            
            try
                s = datasets.record.load_csvfile('../data/test/wine/wine.csv','%d','%s%s%f%f%f%f%f%f%f%f%f%f%f');
                assert(false);
            catch exp
                if strcmp(exp.message,'File "../data/test/wine/wine.csv" has an invalid format!')
                    fprintf('      Passes "Invalid format!" test.\n');
                else
                    assert(false);
                end
            end
            
            clearvars -except display;
        end
    end
end
