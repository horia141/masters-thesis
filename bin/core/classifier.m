classdef classifier
    properties (GetAccess=public,SetAccess=immutable)
        input_geometry;
        saved_labels;
        saved_labels_count;
    end
    
    methods (Access=public)
        function [obj] = classifier(input_geometry,saved_labels)
            assert(check.vector(input_geometry));
            assert((length(input_geometry) == 1) || (length(input_geometry) == 4));
            assert(check.natural(input_geometry));
            assert(check.checkv(input_geometry >= 1));
            assert(check.vector(saved_labels));
            assert(check.cell(saved_labels));
            assert(check.checkf(@check.scalar,saved_labels));
            assert(check.checkf(@check.string,saved_labels));

            obj.input_geometry = input_geometry;
            obj.saved_labels = saved_labels;
            obj.saved_labels_count = length(saved_labels);
        end

        function [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = classify(obj,sample,class_info)
            assert(check.scalar(obj));
            assert(check.classifier(obj));
            assert(check.dataset(sample));
            assert(check.scalar(class_info));
            assert(check.classifier_info(class_info) || (class_info == -1));
            assert(dataset.geom_compatible(obj.input_geometry,dataset.geometry(sample)));
            assert(~check.classifier_info(class_info) || (check.same(obj.saved_labels,class_info.labels)));
            assert(~check.classifier_info(class_info) || class_info.compatible(sample));

            [labels_idx_hat,labels_confidence] = obj.do_classify(sample);
            
            if check.classifier_info(class_info)
                score = 100 * sum(class_info.labels_idx == labels_idx_hat) / length(class_info.labels_idx);
                conf_matrix = confusionmat(class_info.labels_idx,labels_idx_hat);
                misclassified = find(labels_idx_hat ~= class_info.labels_idx);
            else
                score = -1;
                conf_matrix = -1;
                misclassified = -1;
            end
        end
    end
    
    methods (Abstract,Access=protected)
        do_classify(obj,sample);
    end
end
