classdef dataset
    methods (Static,Access=public)
        function [sample,varargout] = load(dataset_path)
            assert(check.scalar(dataset_path));
            assert(check.string(dataset_path));
            
            try
                file_contents = load(dataset_path);
            catch e
                throw(MException('master:NoLoad',e.message));
            end
                
            if ~isfield(file_contents,'PROTOCOL_VERSION') || ...
               ~isfield(file_contents,'PROTOCOL_TYPE') || ...
               ~isfield(file_contents,'PROTOCOL_TYPE_VERSION')
                throw(MException('master:NoLoad','Invalid format!'));
            end
            
            protocol_version = file_contents.PROTOCOL_VERSION;
            protocol_type = file_contents.PROTOCOL_TYPE;
            protocol_type_version = file_contents.PROTOCOL_TYPE_VERSION;
            
            if protocol_version == 1
                if strcmp(protocol_type,'Record+ClassifierInfo')
                    if protocol_type_version == 1
                        if ~isfield(file_contents,'sample') || ...
                           ~isfield(file_contents,'classifier_labels') || ...
                           ~isfield(file_contents,'classifier_idxs')
                            throw(MException('master:NoLoad','Invalid "Record+ClassifierInfo" format!'));
                        end
                        
                        sample = file_contents.sample;
                        classifier_labels = file_contents.classifier_labels;
                        classifier_idxs = file_contents.classifier_idxs;
                        
                        if ~check.dataset_record(sample)
                            throw(MException('master:NoLoad','Invalid sample!'));
                        end
                        
                        if ~(check.vector(classifier_labels) && check.cell(classifier_labels) && ...
                             check.checkf(@check.scalar,classifier_labels) && check.checkf(@check.string,classifier_labels))
                            throw(MException('master:NoLoad','Invalid classifier labels!'));
                        end
                        
                        if ~(check.vector(classifier_idxs) && check.natural(classifier_idxs) && ...
                             check.checkv(classifier_idxs >= 1 & classifier_idxs <= length(classifier_labels)))
                            throw(MException('master:NoLoad','Invalid classifier indices!'));
                        end
                        
                        varargout{1} = classifier_info(classifier_labels,classifier_idxs);
                        
                        if ~varargout{1}.compatible(sample)
                            throw(MException('master:NoLoad','Classifier info not compatible with sample!'));
                        end
                    else
                        throw(MException('master:NoLoad','Unknown "Record+ClassifierInfo" version!'));
                    end
                elseif strcmp(protocol_type,'Record')
                    if ~isfield(file_contents,'sample')
                        throw(MException('master:NoLoad','Invalid "Record" format!'));
                    end
                    
                    sample = file_contents.sample;
                    
                    if ~check.dataset_record(sample)
                        throw(MException('master:NoLoad','Invalid sample!'));
                    end
                elseif strcmp(protocol_type,'Image+ClassifierInfo')
                    if protocol_type_version == 1
                        if ~isfield(file_contents,'sample') || ...
                           ~isfield(file_contents,'classifier_labels') || ...
                           ~isfield(file_contents,'classifier_idxs')
                            throw(MException('master:NoLoad','Invalid "Image+ClassifierInfo" format!'));
                        end
                        
                        sample = file_contents.sample;
                        classifier_labels = file_contents.classifier_labels;
                        classifier_idxs = file_contents.classifier_idxs;
                        
                        if ~check.dataset_image(sample)
                            throw(MException('master:NoLoad','Invalid sample!'));
                        end
                        
                        if ~(check.vector(classifier_labels) && check.cell(classifier_labels) && ...
                             check.checkf(@check.scalar,classifier_labels) && check.checkf(@check.string,classifier_labels))
                            throw(MException('master:NoLoad','Invalid classifier labels!'));
                        end
                        
                        if ~(check.vector(classifier_idxs) && check.natural(classifier_idxs) && ...
                             check.checkv(classifier_idxs >= 1 & classifier_idxs <= length(classifier_labels)))
                            throw(MException('master:NoLoad','Invalid classifier indices!'));
                        end
                        
                        varargout{1} = classifier_info(classifier_labels,classifier_idxs);
                        
                        if ~varargout{1}.compatible(sample)
                            throw(MException('master:NoLoad','Classifier info not compatible with sample!'));
                        end
                    else
                        throw(MException('master:NoLoad','Unknown "Image+ClassifierInfo" version!'));
                    end
                elseif strcmp(protocol_type,'Image')
                    if ~isfield(file_contents,'sample')
                        throw(MException('master:NoLoad','Invalid "Image" format!'));
                    end
                    
                    sample = file_contents.sample;
                    
                    if ~check.dataset_image(sample)
                        throw(MException('master:NoLoad','Invalid sample!'));
                    end
                else
                    throw(MException('master:NoLoad','Unknown protocol type!'));
                end
            else
                throw(MException('master:NoLoad','Unknown protocol version!'));
            end
        end
        
        function [sample_count] = count(dataset)
            assert(check.dataset(dataset));
            
            if check.dataset_record(dataset)
                sample_count = size(dataset,2);
            elseif check.dataset_image(dataset)
                sample_count = size(dataset,4);
            else
                assert(false);
            end
        end
        
        function [varargout] = geometry(dataset)
            assert(check.dataset(dataset));

            if check.dataset_record(dataset)
                varargout{1} = size(dataset,1);
            elseif check.dataset_image(dataset)
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
            assert(check.vector(geom_1));
            assert((length(geom_1) == 1) || (length(geom_1) == 4));
            assert(check.natural(geom_1));
            assert(check.checkv(geom_1 >= 1));
            assert(check.vector(geom_2));
            assert((length(geom_2) == 1) || (length(geom_2) == 4));
            assert(check.natural(geom_2));
            assert(check.checkv(geom_2 >= 1));
            
            o = check.same(geom_1,geom_2);
        end
        
        function [new_sample] = rebuild_image(sample,layers_count,row_count,col_count)
            assert(check.dataset_record(sample));
            assert(check.scalar(layers_count));
            assert(check.natural(layers_count));
            assert(layers_count >= 1);
            assert(check.scalar(row_count));
            assert(check.natural(row_count));
            assert(row_count >= 1);
            assert(check.scalar(col_count));
            assert(check.natural(col_count));
            assert(col_count >= 1);
            assert(size(sample,1) == (layers_count * row_count * col_count));
            
            N = dataset.count(sample);
            new_sample = reshape(sample,row_count,col_count,layers_count,N);
        end

        function [new_sample] = flatten_image(sample)
            assert(check.dataset_image(sample));
            
            N = dataset.count(sample);
            [d,~,~,~] = dataset.geometry(sample);
            new_sample = reshape(sample,d,N);
        end

        function [new_sample] = subsample(sample,index)
            assert(check.dataset(sample));
            assert(check.vector(index));
            
            if check.dataset_record(sample)
                assert((check.logical(index) && check.match_dims(sample,index,2,2)) || ...
                       (check.natural(index) && check.checkv(index >= 1 & index <= size(sample,2))));
                   
                new_sample = sample(:,index);
            elseif check.dataset_image(sample)
                assert((check.logical(index) && check.match_dims(sample,index,4)) || ...
                       (check.natural(index) && check.checkv(index >= 1 & index <= size(sample,4))));
                   
                new_sample = sample(:,:,:,index);
            else
                assert(false);
            end
        end
    end

    methods (Static,Access=public)
        function test(~)
            fprintf('Testing "dataset".\n');
            
            fprintf('  Function "load".\n');
            
            fprintf('    With proper record and classifier info data.\n');
            
            [s,ci] = dataset.load('../../test/good_record_classifier_info.mat');
            
            assert(check.same(s,[1 0; 1 1; 0 1]'));
            assert(check.same(ci.labels,{'1' '2' '3'}));
            assert(check.same(ci.labels_count,3));
            assert(check.same(ci.labels_idx,[3 2 1]));
            
            clearvars -except test_figure;
            
            fprintf('    With proper record data.\n');
            
            s = dataset.load('../../test/good_record.mat');
            
            assert(check.same(s,[1 0; 1 1; 0 1]'));
            
            clearvars -except test_figure;
            
            fprintf('    With proper image and classifier info data.\n');
            
            [s,ci] = dataset.load('../../test/good_image_classifier_info.mat');
            
            assert(check.same(s,cat(4,[1 0; 0 0],[0 1; 0 0],[0 0; 1 0],[0 0; 0 1])));
            assert(check.same(ci.labels,{'1' '2'}));
            assert(check.same(ci.labels_count,2));
            assert(check.same(ci.labels_idx,[1 1 2 2]));
            
            clearvars -except test_figure;
            
            fprintf('    With proper image data.\n');
            
            s = dataset.load('../../test/good_image.mat');
            
            assert(check.same(s,cat(4,[1 0; 0 0],[0 1; 0 0],[0 0; 1 0],[0 0; 0 1])));
            
            clearvars -except test_figure;
            
            fprintf('    With improper external inputs.\n');
            
            try
                dataset.load('../../test/bad_format.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
            end
            
            try
                dataset.load('../../test/bad_no_protocol_version.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid format!'));
            end
            
            try
                dataset.load('../../test/bad_no_protocol_type.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid format!'));
            end
            
            try
                dataset.load('../../test/bad_no_protocol_type_version.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid format!'));
            end
            
            try
                dataset.load('../../test/bad_protocol_version.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Unknown protocol version!'));
            end
            
            try
                dataset.load('../../test/bad_protocol_type.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Unknown protocol type!'));
            end
            
            try
                dataset.load('../../test/bad_record_classifier_info_version.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Unknown "Record+ClassifierInfo" version!'));
            end
            
            try
                dataset.load('../../test/bad_record_classifier_info_no_sample.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid "Record+ClassifierInfo" format!'));
            end
            
            try
                dataset.load('../../test/bad_record_classifier_info_no_classifier_labels.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid "Record+ClassifierInfo" format!'));
            end
            
            try
                dataset.load('../../test/bad_record_classifier_info_no_classifier_idxs.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid "Record+ClassifierInfo" format!'));
            end 
            
            try
                dataset.load('../../test/bad_record_classifier_info_sample.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid sample!'));
            end
            
            try
                dataset.load('../../test/bad_record_classifier_info_classifier_labels.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid classifier labels!'));
            end
            
            try
                dataset.load('../../test/bad_record_classifier_info_classifier_idxs.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid classifier indices!'));
            end
            
            try
                dataset.load('../../test/bad_record_classifier_info_incompatible.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Classifier info not compatible with sample!'));
            end
            
            try
                dataset.load('../../test/bad_record_no_sample.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid "Record" format!'));
            end
            
            try
                dataset.load('../../test/bad_record_sample.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid sample!'));
            end
            
            try
                dataset.load('../../test/bad_image_classifier_info_version.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Unknown "Image+ClassifierInfo" version!'));
            end
            
            try
                dataset.load('../../test/bad_image_classifier_info_no_sample.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid "Image+ClassifierInfo" format!'));
            end
            
            try
                dataset.load('../../test/bad_image_classifier_info_no_classifier_labels.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid "Image+ClassifierInfo" format!'));
            end
            
            try
                dataset.load('../../test/bad_image_classifier_info_no_classifier_idxs.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid "Image+ClassifierInfo" format!'));
            end 
            
            try
                dataset.load('../../test/bad_image_classifier_info_sample.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid sample!'));
            end
            
            try
                dataset.load('../../test/bad_image_classifier_info_classifier_labels.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid classifier labels!'));
            end
            
            try
                dataset.load('../../test/bad_image_classifier_info_classifier_idxs.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid classifier indices!'));
            end
            
            try
                dataset.load('../../test/bad_image_classifier_info_incompatible.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Classifier info not compatible with sample!'));
            end
            
            try
                dataset.load('../../test/bad_image_no_sample.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid "Image" format!'));
            end
            
            try
                dataset.load('../../test/bad_image_sample.mat');
                assert(false);
            catch e
                assert(check.same(e.identifier,'master:NoLoad'));
                assert(check.same(e.message,'Invalid sample!'));
            end
            
            fprintf('  Function "count".\n');
            
            s_1 = randi(2,4,10);
            s_2 = randi(2,8,8,1,10);
            
            assert(dataset.count(s_1) == 10);
            assert(dataset.count(s_2) == 10);
            
            clearvars -except test_figure;
            
            fprintf('  Function "geometry".\n');
            
            fprintf('    Geometry of records.\n');
            
            s = randi(2,4,10);
            
            d = dataset.geometry(s);
            
            assert(d == 4);
            
            clearvars -except test_figure;
            
            fprintf('    Geometry of images.\n');
            
            s = randi(2,8,4,3,100);
            
            g = dataset.geometry(s);
            
            assert(check.same(g,[8*4*3 8 4 3]));
            
            [d,d_r,d_c,d_l] = dataset.geometry(s);
            
            assert(d == 8*4*3);
            assert(d_r == 8);
            assert(d_c == 4);
            assert(d_l == 3);
            
            clearvars -except test_figure;
            
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
            
            clearvars -except test_figure;
            
            fprintf('  Function "rebuild_image".\n');
            
            fprintf('    Single layer.\n');
            
            A = rand(100,20);
            A_i = zeros(10,10,1,20);
            for ii = 1:20
                A_i(:,:,1,ii) = reshape(A(:,ii),[10 10]);
            end
            
            s = dataset.rebuild_image(A,1,10,10);
            
            assert(check.dataset_image(s));
            assert(check.same(size(s),[10 10 1 20]));
            assert(check.unitreal(s));
            assert(check.same(s,A_i));
            
            clearvars -except test_figure;
            
            fprintf('    Three layers.\n');
            
            A = rand(300,20);
            A_i = zeros(10,10,3,20);
            for ii = 1:20
                A_i(:,:,:,ii) = reshape(A(:,ii),[10 10 3]);
            end
            
            s = dataset.rebuild_image(A,3,10,10);
            
            assert(check.dataset_image(s));
            assert(check.same(size(s),[10 10 3 20]));
            assert(check.unitreal(s));
            assert(check.same(s,A_i));

            clearvars -except test_figure;
            
            fprintf('  Function "flatten_image".\n');
            
            fprintf('    Single layer.\n');
            
            A = rand(10,10,1,20);
            A_i = zeros(100,20);
            for ii = 1:20
                A_i(:,ii) = reshape(A(:,:,:,ii),100,1);
            end
            
            s = dataset.flatten_image(A);
            
            assert(check.dataset_record(s));
            assert(check.same(size(s),[100 20]));
            assert(check.unitreal(s));
            assert(check.same(s,A_i));
            
            clearvars -except test_figure;
            
            fprintf('    Three layers.\n');
            
            A = rand(10,10,3,20);
            A_i = zeros(300,20);
            for ii = 1:20
                A_i(:,ii) = reshape(A(:,:,:,ii),300,1);
            end
            
            s = dataset.flatten_image(A);
            
            assert(check.dataset_record(s));
            assert(check.same(size(s),[300 20]));
            assert(check.unitreal(s));
            assert(check.same(s,A_i));
            
            clearvars -except test_figure;
            
            fprintf('  Function "subsample".\n');
            
            fprintf('    With boolean indices on records.\n');
            
            s = rand(10,100);
            idx = logical(randi(2,1,100) - 1);
            
            s_1 = dataset.subsample(s,idx);
            
            assert(check.dataset_record(s_1));
            assert(check.same(size(s_1),[10 sum(idx)]));
            assert(check.unitreal(s_1));
            assert(check.same(s_1,s(:,idx)));
            
            clearvars -except test_figure;
            
            fprintf('    With boolean indices on images.\n');
            
            s = rand(8,8,3,100);
            idx = logical(randi(2,1,100) - 1);
            
            s_1 = dataset.subsample(s,idx);
            
            assert(check.dataset_image(s_1));
            assert(check.same(size(s_1),[8 8 3 sum(idx)]));
            assert(check.unitreal(s_1));
            assert(check.same(s_1,s(:,:,:,idx)));
            
            clearvars -except test_figure;
            
            fprintf('    With integer indices on records.\n');
            
            s = rand(10,100);
            idx = randi(100,20,1);
            
            s_1 = dataset.subsample(s,idx);
            
            assert(check.dataset_record(s_1));
            assert(check.same(size(s_1),[10 20]));
            assert(check.unitreal(s_1));
            assert(check.same(s_1,s(:,idx)));
            
            clearvars -except test_figure;
            
            fprintf('    With integer indices on images.\n');
            
            s = rand(8,8,3,100);
            idx = randi(100,10,1);
            
            s_1 = dataset.subsample(s,idx);
            
            assert(check.dataset_image(s_1));
            assert(check.same(size(s_1),[8 8 3 10]));
            assert(check.unitreal(s_1));
            assert(check.same(s_1,s(:,:,:,idx)));
            
            clearvars -except test_figure;
        end
    end
end
