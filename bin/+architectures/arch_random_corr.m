classdef arch_random_corr < architecture
    properties (GetAccess=public,SetAccess=immutable)
        filters_count;
        filter_row_count;
        filter_col_count;
        reduce_function;
        reduce_spread;
    end
    
    methods (Access=public)
        function [obj] = arch_random_corr(train_sample_plain,filters_count,filter_row_count,filter_col_count,reduce_function,reduce_spread,classifier_ctor_fn,classifier_params,logger)
            assert(tc.dataset_image(train_sample_plain));
            assert(size(train_sample_plain,3) == 1); % A BIT OF A HACK
            assert(tc.scalar(filters_count));
            assert(tc.natural(filters_count));
            assert(filters_count >= 1);
            assert(tc.scalar(filter_row_count));
            assert(tc.natural(filter_row_count));
            assert(filter_row_count >= 1);
            assert(mod(filter_row_count,2) == 1);
            assert(tc.scalar(filter_col_count));
            assert(tc.natural(filter_col_count));
            assert(filter_col_count >= 1);
            assert(mod(filter_col_count,2) == 1);
            assert(tc.scalar(reduce_function));
            assert(tc.function_h(reduce_function));
            assert(tc.scalar(reduce_spread));
            assert(tc.natural(reduce_spread));
            assert(reduce_spread >= 1);
            assert(tc.scalar(classifier_ctor_fn));
            assert(tc.function_h(classifier_ctor_fn));
            assert(tc.empty(classifier_params) || tc.vector(classifier_params));
            assert(tc.empty(classifier_params) || tc.cell(classifier_params));
            assert(tc.empty(classifier_params) || tc.checkf(@tc.value,classifier_params));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(mod(train_sample_plain.row_count - filter_row_count + 1,reduce_spread) == 0);
            assert(mod(train_sample_plain.col_count - filter_col_count + 1,reduce_spread) == 0);
            
            t_random_corr = transforms.image.random_corr(train_sample_plain,filters_count,filter_row_count,filter_col_count,...
                                                         reduce_function,reduce_spread,logger.new_transform('Training RandomCorr transform'));
            train_image_1 = t_random_corr.code(train_sample_plain,logger.new_transform('Transforming training dataset'));

            classifier = classifier_ctor_fn(train_image_1,classifier_params{:},logger.new_classifier('Training classifier'));
            
            obj = obj@architecture(train_sample_plain.subsamples(1),{t_random_corr},classifier,logger);
            obj.filters_count = filters_count;
            obj.filter_row_count = filter_row_count;
            obj.filter_col_count = filter_col_count;
            obj.reduce_function = reduce_function;
            obj.reduce_spread = reduce_spread;
        end
    end
end
