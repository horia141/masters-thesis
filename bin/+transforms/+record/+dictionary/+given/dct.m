classdef dct < transforms.record.dictionary
    properties (GetAccess=public,SetAccess=immutable)
        geometry_explicit;
        word_count_explicit;
    end
    
    methods (Access=public)
        function [obj] = dct(train_sample_plain,word_count,coding_method,coding_params,coeff_count,num_workers,geometry_explicit)
            assert(check.dataset_record(train_sample_plain));
            assert(check.vector(word_count));
            assert((length(word_count) == 1) || ...
                   (length(word_count) == 2));
            assert(check.natural(word_count));
            assert(check.checkv(word_count >= 1));
            assert(prod(word_count) <= dataset.geometry(train_sample_plain));
            assert(transforms.record.dictionary.coding_setup_ok(coding_method,coding_params));
            assert(check.scalar(coeff_count));
            assert(check.natural(coeff_count));
            assert(coeff_count >= 1);
            assert(coeff_count <= prod(word_count));
            assert(check.scalar(num_workers));
            assert(check.natural(num_workers));
            assert(num_workers >= 1);
            assert(check.vector(geometry_explicit));
            assert((length(geometry_explicit) == 1) || ...
                   (length(geometry_explicit) == 2));
            assert(check.natural(geometry_explicit));
            assert(check.checkv(geometry_explicit >= 1));
            assert(prod(geometry_explicit) == dataset.geometry(train_sample_plain));
            
            d = dataset.geometry(train_sample_plain);
            
            if length(word_count) == 1
                dict = zeros(word_count,d);
                
                for freq = 1:word_count
                    dict(freq,:) = cos(pi/d * (0:(d-1) + 1/2) * (freq - 1));
                end
            elseif length(word_count) == 2
                dict_transp = zeros(geometry_explicit(1),geometry_explicit(2),word_count(1),word_count(2));
                
                for freq1 = 1:word_count(1)
                    for freq2 = 1:word_count(2)
                        for ii = 1:geometry_explicit(1)
                            for jj = 1:geometry_explicit(2)
                               dict_transp(ii,jj,freq1,freq2) = cos(pi/geometry_explicit(1) * (ii + 1/2) * (freq1 - 1)) * cos(pi/geometry_explicit(2) * (jj + 1/2) * (freq2 - 1));
                            end
                        end
                    end
                end
                
                dict = reshape(dict_transp,prod(geometry_explicit),prod(word_count))';
            else
                assert(false);
            end
            
            obj = obj@transforms.record.dictionary(train_sample_plain,dict,coding_method,coding_params,coeff_count,num_workers);
            obj.geometry_explicit = geometry_explicit;
            obj.word_count_explicit = word_count;
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.dictionary.given.dct".\n');
            
            fprintf('  Proper constuction.\n');
            
            fprintf('    NOT YET TESTED.\n');
        end
    end
end

