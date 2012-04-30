classdef classifier
    properties (GetAccess=public,SetAccess=immutable)
        input_geometry;
        saved_labels;
        saved_labels_count;
    end
    
    methods (Access=public)
        function [obj] = classifier(input_geometry,saved_labels,logger)
            assert(tc.vector(input_geometry));
            assert((length(input_geometry) == 1) || (length(input_geometry) == 4));
            assert(tc.natural(input_geometry));
            assert(tc.check(input_geometry >= 1));
            assert(tc.vector(saved_labels));
            assert(tc.labels(saved_labels));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);

            obj.input_geometry = input_geometry;
            obj.saved_labels = saved_labels;
            obj.saved_labels_count = length(saved_labels);
        end

        function [labels_idx_hat,labels_confidence,score,conf_matrix,misclassified] = classify(obj,sample,class_info,logger)
            assert(tc.scalar(obj));
            assert(tc.classifier(obj));
            assert(tc.dataset(sample));
            assert(tc.scalar(class_info));
            assert(tc.classification_info(class_info) || (class_info == -1));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(dataset.geom_compatible(obj.input_geometry,dataset.geometry(sample)));
            assert(~tc.classification_info(class_info) || (tc.same(obj.saved_labels,class_info.labels)));
            assert(~tc.classification_info(class_info) || class_info.compatible(sample));

            [labels_idx_hat,labels_confidence] = obj.do_classify(sample,logger);
            
            if tc.classification_info(class_info)
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
        do_classify(obj,sample,logger);
    end
end
