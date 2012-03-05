classdef samples_set
    properties (GetAccess=public,SetAccess=immutable)
        classes;
        samples;
        labels_idx;
    end
    
    properties (GetAccess=public,SetAccess=private,Dependent=true)
        classes_count;
        samples_count;
        features_count;
    end
    
    methods
        function [classes_count] = get.classes_count(obj)
            classes_count = length(obj.classes);
        end
        
        function [count] = get.samples_count(obj)
            count = size(obj.samples,1);
        end
        
        function [features_count] = get.features_count(obj)
            features_count = size(obj.samples,2);
        end
    end
    
    methods (Access=public)
        function [obj] = samples_set(classes,samples,labels_idx)
            assert(tc.vector(classes) && tc.cell(classes));
            assert(tc.matrix(samples) && tc.number(samples));
            assert(tc.vector(labels_idx) && tc.match_dims(samples,labels_idx,1) && ...
                   tc.labels_idx(labels_idx,classes));

            obj.classes = utils.force_col(classes);
            obj.samples = samples;
            obj.labels_idx = utils.force_col(labels_idx);
        end
        
        function [tr_index,ts_index] = partition(obj,type,param)
            assert(tc.string(type) && (strcmp(type,'kfold') || strcmp(type,'holdout')));
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
        
        function [new_samples_set] = subsamples(obj,index)
            assert(tc.vector(index) && ...
                   ((tc.logical(index) && tc.match_dims(obj.samples,index,1)) || ...
                    (tc.natural(index) && tc.check(index > 0 & index <= obj.samples_count))));
            
            new_samples_set = samples_set(obj.classes,obj.samples(index,:),obj.labels_idx(index));
        end
    end
    
    methods (Static,Access=public)
        function [new_samples_set] = from_data(samples,labels)
            assert(tc.matrix(samples) && tc.number(samples));
            assert(tc.vector(labels) && tc.match_dims(samples,labels,1) && tc.labels(labels));
            
            [labels_idx_t,classes_t] = grp2idx(labels);
            new_samples_set = samples_set(classes_t,samples,labels_idx_t);
        end
        
        function [new_samples_set] = load_csvfile(csvfile_path,labels_format,data_format,delimiter)
            assert(tc.string(csvfile_path));
            assert(tc.string(labels_format));
            assert(tc.string(data_format));
            assert(~exist('delimiter','var') || tc.string(delimiter));
            
            if ~exist('delimiter','var')
                delimiter = ',';
            end
            
            try
                [csvfile_fid,csvfile_msg] = fopen(csvfile_path,'rt');
                
                if csvfile_fid == -1
                    throw(MException('master:Samples:load_csvfile:NoLoad',...
                             sprintf('Could not load csv file "%s": %s!',csvfile_path,csvfile_msg)))
                end
                
                samples_raw = textscan(csvfile_fid,strcat(labels_format,data_format),'delimiter',delimiter);
                fclose(csvfile_fid);
            catch exp
                throw(MException('master:Samples:load_csvfile:NoLoad',exp.message));
            end
            
            if ~all(cellfun(@tc.number,samples_raw(:,2:end)))
                throw(MException('master:Samples:load_csvfile:InvalidFormat',...
                         sprintf('File "%s" has an invalid format!',csvfile_path)));
            end
            
            samples_t = cell2mat(samples_raw(:,2:end));
            labels_t = samples_raw{:,1};
            
            new_samples_set = samples_set.from_data(samples_t,labels_t);
        end
    end
    
    methods (Static,Access=public)
        function test
            fprintf('Testing "samples_set".\n');
            
            % Try a normal run first. Just build an object using the
            % constructor and call the two possible methods ("partition"
            % and "subsamples"). In all cases, see if we obtain correct
            % results, that is the internal representation matches what we 
            % would expect, given our data.
            
            fprintf('  Testing proper construction, "partition" and "subsamples".\n');
            
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
             
            s1 = samples_set({'1' '2' '3'},A,c);
            
            assert(length(s1.classes) == 3);
            assert(strcmp(s1.classes(1),'1'));
            assert(strcmp(s1.classes(2),'2'));
            assert(strcmp(s1.classes(3),'3'));
            assert(s1.classes_count == 3);
            assert(all(size(s1.samples) == [12 4]));
            assert(all(all(s1.samples == A)));
            assert(length(s1.labels_idx) == 12);
            assert(all(s1.labels_idx == c'));
            assert(s1.samples_count == 12);
            assert(s1.features_count == 4);
            
            [tr_f,ts_f] = s1.partition('kfold',2);
            
            s1_f11 = s1.subsamples(tr_f(:,1));
            
            assert(length(s1_f11.classes) == 3);
            assert(strcmp(s1_f11.classes(1),'1'));
            assert(strcmp(s1_f11.classes(2),'2'));
            assert(strcmp(s1_f11.classes(3),'3'));
            assert(s1_f11.classes_count == 3);
            assert(all(size(s1_f11.samples) == [6 4]));
            assert(all(all(s1_f11.samples == A(tr_f(:,1),:))));
            assert(length(s1_f11.labels_idx) == 6);
            assert(all(s1_f11.labels_idx == c(tr_f(:,1))'));
            assert(s1_f11.samples_count == 6);
            assert(s1_f11.features_count == 4);
            
            clear s1_f11
            
            s1_f12 = s1.subsamples(ts_f(:,1));
            
            assert(length(s1_f12.classes) == 3);
            assert(strcmp(s1_f12.classes(1),'1'));
            assert(strcmp(s1_f12.classes(2),'2'));
            assert(strcmp(s1_f12.classes(3),'3'));
            assert(s1_f12.classes_count == 3);
            assert(all(size(s1_f12.samples) == [6 4]));
            assert(all(all(s1_f12.samples == A(ts_f(:,1),:))));
            assert(length(s1_f12.labels_idx) == 6);
            assert(all(s1_f12.labels_idx == c(ts_f(:,1))'));
            assert(s1_f12.samples_count == 6);
            assert(s1_f12.features_count == 4);
            
            clear s1_f12
            
            s1_f21 = s1.subsamples(tr_f(:,2));

            assert(length(s1_f21.classes) == 3);
            assert(strcmp(s1_f21.classes(1),'1'));
            assert(strcmp(s1_f21.classes(2),'2'));
            assert(strcmp(s1_f21.classes(3),'3'));
            assert(s1_f21.classes_count == 3);
            assert(all(size(s1_f21.samples) == [6 4]));
            assert(all(all(s1_f21.samples == A(tr_f(:,2),:))));
            assert(length(s1_f21.labels_idx) == 6);
            assert(all(s1_f21.labels_idx == c(tr_f(:,2))'));
            assert(s1_f21.samples_count == 6);
            assert(s1_f21.features_count == 4);
            
            clear s1_f21
            
            s1_f22 = s1.subsamples(ts_f(:,2));
            
            assert(length(s1_f22.classes) == 3);
            assert(strcmp(s1_f22.classes(1),'1'));
            assert(strcmp(s1_f22.classes(2),'2'));
            assert(strcmp(s1_f22.classes(3),'3'));
            assert(s1_f22.classes_count == 3);
            assert(all(size(s1_f22.samples) == [6 4]));
            assert(all(all(s1_f22.samples == A(ts_f(:,2),:))));
            assert(length(s1_f22.labels_idx) == 6);
            assert(all(s1_f22.labels_idx == c(ts_f(:,2))'));
            assert(s1_f22.samples_count == 6);
            assert(s1_f22.features_count == 4);
            
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
            assert(all(size(s1_h1.samples) == [9 4]));
            assert(all(all(s1_h1.samples == A(tr_h,:))));
            assert(length(s1_h1.labels_idx) == 9);
            assert(all(s1_h1.labels_idx == c(tr_h)'));
            assert(s1_h1.samples_count == 9);
            assert(s1_h1.features_count == 4);
            
            clear s1_h1
            
            s1_h2 = s1.subsamples(ts_h);
            
            assert(length(s1_h2.classes) == 3);
            assert(strcmp(s1_h2.classes(1),'1'));
            assert(strcmp(s1_h2.classes(2),'2'));
            assert(strcmp(s1_h2.classes(2),'2'));
            assert(s1_h2.classes_count == 3);
            assert(all(size(s1_h2.samples) == [3 4]));
            assert(all(all(s1_h2.samples == A(ts_h,:))));
            assert(length(s1_h2.labels_idx) == 3);
            assert(all(s1_h2.labels_idx == c(ts_h)'));
            assert(s1_h2.samples_count == 3);
            assert(s1_h2.features_count == 4);
            
            clear s1_h2
            clear tr_h
            clear ts_h
            
            s1_fi = s1.subsamples([1:2:12]);
            
            assert(length(s1_fi.classes) == 3);
            assert(strcmp(s1_fi.classes(1),'1'));
            assert(strcmp(s1_fi.classes(2),'2'));
            assert(strcmp(s1_fi.classes(3),'3'));
            assert(s1_fi.classes_count == 3);
            assert(all(size(s1_fi.samples) == [6 4]));
            assert(all(all(s1_fi.samples == A(1:2:12,:))));
            assert(length(s1_fi.labels_idx) == 6);
            assert(all(s1_fi.labels_idx == c(1:2:12)'));
            assert(s1_fi.samples_count == 6);
            assert(s1_fi.features_count == 4);
            
            clear s1_fi
            
            s1_fo = s1.subsamples([1:12,1:12]);
            
            assert(length(s1_fo.classes) == 3);
            assert(strcmp(s1_fo.classes(1),'1'));
            assert(strcmp(s1_fo.classes(2),'2'));
            assert(strcmp(s1_fo.classes(3),'3'));
            assert(s1_fo.classes_count == 3);
            assert(all(size(s1_fo.samples) == [24 4]));
            assert(all(all(s1_fo.samples == [A;A])));
            assert(length(s1_fo.labels_idx) == 24);
            assert(all(s1_fo.labels_idx == [c c]'));
            assert(s1_fo.samples_count == 24);
            assert(s1_fo.features_count == 4);
            
            clear s1_fo
            
            clear s1
            
            % Try building from pre-existing data using the "from_data"
            % static method.
            
            fprintf('  Testing "from_data".\n');
            
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
            
            s2 = samples_set.from_data(A,c);
            
            assert(length(s2.classes) == 3);
            assert(strcmp(s2.classes(1),'1'));
            assert(strcmp(s2.classes(2),'2'));
            assert(strcmp(s2.classes(3),'3'));
            assert(s2.classes_count == 3);
            assert(all(size(s2.samples) == [12 4]));
            assert(all(all(s2.samples == A)));
            assert(length(s2.labels_idx) == 12);
            assert(all(s2.labels_idx == c'));
            assert(s2.samples_count == 12);
            assert(s2.features_count == 4);
            
            clear s2
            
            % Try loading from an CSV file. The file we're going to use is
            % "$PROJECT_ROOT/data/iris/iris.csv". This should exist in all
            % distributions of this project.
            
            fprintf('  Testing "load_csvfile" with iris data.\n');
            
            s3 = samples_set.load_csvfile('../data/iris/iris.csv','%s','%f%f%f%f',',');
            
            assert(length(s3.classes) == 3);
            assert(strcmp(s3.classes(1),'Iris-setosa'));
            assert(strcmp(s3.classes(2),'Iris-versicolor'));
            assert(strcmp(s3.classes(3),'Iris-virginica'));
            assert(s3.classes_count == 3);
            assert(all(size(s3.samples) == [150 4]));
            assert(tc.matrix(s3.samples) && tc.number(s3.samples));
            assert(all(size(s3.labels_idx) == [150 1]));
            assert(all(s3.labels_idx == [1*ones(50,1);2*ones(50,1);3*ones(50,1)]));
            assert(s3.samples_count == 150);
            assert(s3.features_count == 4);
            
            clear s3
            
            % Try loading from another CSV file. The file we're going to
            % use is "$PROJECT_ROOT/data/wine/wine.csv". This should exist
            % in all distributions of this project. We're not going to
            % specify a delimiter here.
            
            fprintf('  Testing "load_csvfile" with Wine data.\n');
            
            s4 = samples_set.load_csvfile('../data/wine/wine.csv','%d','%f%f%f%f%f%f%f%f%f%f%f%f%f');
            
            assert(length(s4.classes) == 3);
            assert(strcmp(s4.classes(1),'1'));
            assert(strcmp(s4.classes(2),'2'));
            assert(strcmp(s4.classes(3),'3'));
            assert(s4.classes_count == 3);
            assert(all(size(s4.samples) == [178 13]));
            assert(tc.matrix(s4.samples) && tc.number(s4.samples));
            assert(all(size(s4.labels_idx) == [178 1]));
            assert(all(s4.labels_idx == [1*ones(59,1);2*ones(71,1);3*ones(48,1)]));
            assert(s4.samples_count == 178);
            assert(s4.features_count == 13);
            
            clear s4
            
            % Try some invalid calls to "load_csvfile". These test the
            % failure modes of this function. We're interested in things
            % beyond the caller's control like improperly formated files or
            % insufficient access rights.
            
            fprintf('  Testing "load_csvfile" with invalid external inputs.\n');
            
            try
                s5 = samples_set.load_csvfile('../data/wine/wine_aaa.csv','%d','%f%f%f%f%f%f%f%f%f%f%f%f%f');
                assert(false);
            catch exp
                if strcmp(exp.message,'Could not load csv file "../data/wine/wine_aaa.csv": No such file or directory!')
                    fprintf('    Passes "No such file or directory!" test.\n');
                else
                    assert(false);
                end
            end
            
            try
                !chmod a-r ../data/wine/wine.csv
                s6 = samples_set.load_csvfile('../data/wine/wine.csv','%d','%f%f%f%f%f%f%f%f%f%f%f%f%f');
                !chmod a+r ../data/wine/wine.csv
                assert(false);
            catch exp
                !chmod a+r ../data/wine/wine.csv
                if strcmp(exp.message,'Could not load csv file "../data/wine/wine.csv": Permission denied!');
                    fprintf('    Passes "Permission denied!" test.\n');
                else
                    assert(false);
                end
            end
            
            try
                s7 = samples_set.load_csvfile('../data/wine/wine.csv','%d','%s%s%f%f%f%f%f%f%f%f%f%f%f');
                assert(false);
            catch exp
                if strcmp(exp.message,'File "../data/wine/wine.csv" has an invalid format!')
                    fprintf('    Passes "Invalid format!" test.\n');
                else
                    assert(false);
                end
            end
        end
    end
end
