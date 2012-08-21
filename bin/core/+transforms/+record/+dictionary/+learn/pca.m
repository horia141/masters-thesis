classdef pca < transforms.record.dictionary
    methods (Access=public)
        function [obj] = pca(train_sample_plain,word_count,coding_method,coding_params,coeff_count,num_workers)
            assert(check.dataset_record(train_sample_plain));
            assert(check.scalar(word_count));
            assert(check.natural(word_count));
            assert(word_count >= 1);
            assert(word_count <= dataset.geometry(train_sample_plain));
            assert(transforms.record.dictionary.coding_setup_ok(coding_method,coding_params));
            assert(check.scalar(coeff_count));
            assert(check.natural(coeff_count));
            assert(coeff_count >= 1);
            assert(coeff_count <= word_count);
            assert(check.scalar(num_workers));
            assert(check.natural(num_workers));
            assert(num_workers >= 1);
            
            coeffs_t = princomp(train_sample_plain');
            dict_t = coeffs_t(:,1:word_count)';
            
            obj = obj@transforms.record.dictionary(train_sample_plain,dict_t,coding_method,coding_params,coeff_count,num_workers);
        end
    end
    
    methods (Static,Access=public)
        function test(test_display)
            fprintf('Testing "transforms.record.dictionary.learn.pca".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    NOT YET TESTED!!!\n');
        end
    end
end
