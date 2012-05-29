classdef random_corr < transform
    properties (GetAccess=public,SetAccess=immutable)
        filters;
        filters_count;
        filter_row_count;
        filter_col_count;
        reduce_fn;
        reduce_spread;
        result_row_count;
        result_col_count;
    end
    
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = random_corr(train_sample_plain,filters_count,filter_row_count,filter_col_count,reduce_fn,reduce_spread,logger)
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
            assert(tc.scalar(reduce_fn));
            assert(tc.function_h(reduce_fn));
            assert(tc.one_of(reduce_fn,@transforms.image.random_corr.sqr,@transforms.image.random_corr.max));
            assert(tc.scalar(reduce_spread));
            assert(tc.natural(reduce_spread));
            assert(reduce_spread >= 1);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert(mod(size(train_sample_plain,1) - filter_row_count + 1,reduce_spread) == 0); % A BIT OF A HACK
            assert(mod(size(train_sample_plain,2) - filter_col_count + 1,reduce_spread) == 0); % A BIT OF A HACK
            
            [d,dr,dc,dl] = dataset.geometry(train_sample_plain);
            
            result_row_count_t = (dr - filter_row_count + 1) / reduce_spread;
            result_col_count_t = (dc - filter_col_count + 1) / reduce_spread;
            
            logger.message('Filter bank size: %dx%d',result_row_count_t,result_col_count_t);
            logger.message('Building filter bank.');
            
            filters_t = 0.1 * rand(filter_row_count,filter_col_count,filters_count) - 0.05;
            
            input_geometry = [d,dr,dc,dl];
            output_geometry = [result_row_count_t * result_col_count_t * filters_count result_row_count_t result_col_count_t filters_count];
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.filters = filters_t;
            obj.filters_count = filters_count;
            obj.filter_row_count = filter_row_count;
            obj.filter_col_count = filter_col_count;
            obj.reduce_fn = reduce_fn;
            obj.reduce_spread = reduce_spread;
            obj.result_row_count = result_row_count_t;
            obj.result_col_count = result_col_count_t;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            N = dataset.count(sample_plain);
            
            sample_coded = zeros(obj.result_row_count,obj.result_col_count,obj.filters_count,N);
            
            logger.beg_node('Applying filters to each image');
            
            for filter = 1:obj.filters_count
                logger.beg_node('Filter #%d',filter);
                
                logger.message('Convolving with filter kernel.');

                conv_images = arrayfun(@(ii)conv2(sample_plain(:,:,1,ii),obj.filters(:,:,filter),'valid'),1:N,'UniformOutput',false);
                conv_images = cat(3,conv_images{:});
                
                logger.message('Reducing resulting images.');
            
                for ii = 1:obj.result_row_count
                    for jj = 1:obj.result_col_count
                        sample_coded(ii,jj,filter,:) = obj.reduce_fn(conv_images(((ii - 1)*obj.reduce_spread + 1):(ii*obj.reduce_spread),...
                                                                                 ((jj - 1)*obj.reduce_spread + 1):(jj*obj.reduce_spread),:));
                    end
                end
                
                logger.end_node();
            end
            
            logger.end_node();
        end
    end
    
    methods (Static,Access=public)
        function [o] = sqr(A)
            o = sum(sum(A .^ 2,1),2);
        end
        
        function [o] = max(A)
            o = max(max(A,[],1),[],2);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.random_corr".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s1 = dataset.load_image_from_dir('../test/scenes_small');
            t1 = transforms.image.patch_extract(s1,20,32,32,0,log);
            s2 = t1.code(s1,log);
            
            t2 = transforms.image.random_corr(s2,4,5,5,@transforms.image.random_corr.sqr,4,log);
            
            assert(tc.same(size(t2.filters),[5 5 4]));
            assert(tc.tensor(t2.filters,3));
            assert(tc.check(t2.filters >= -0.05 & t2.filters <= 0.05));
            assert(t2.filters_count == 4);
            assert(t2.filter_row_count == 5);
            assert(t2.filter_col_count == 5);
            assert(tc.same(t2.reduce_fn,@transforms.image.random_corr.sqr));
            assert(t2.reduce_spread == 4);
            assert(t2.result_row_count == 7);
            assert(t2.result_col_count == 7);
            assert(tc.same(t2.input_geometry,[32*32*1 32 32 1]));
            assert(tc.same(t2.output_geometry,[7*7*4 7 7 4]));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With random patches from "scenes_small".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s1 = dataset.load_image_from_dir('../test/scenes_small');
            t1 = transforms.image.patch_extract(s1,20,32,32,0,log);
            s2 = t1.code(s1,log);
            
            t2 = transforms.image.random_corr(s2,4,5,5,@transforms.image.random_corr.sqr,4,log);
            s2_p = t2.code(s2,log);
            
            assert(tc.tensor(s2_p,4));
            assert(tc.same(size(s2_p),[7 7 4 20]));
            assert(tc.number(s2_p));
            
            if exist('display','var') && (display)
                subplot(5,1,1);
                imshow(utils.format_as_tiles(s2(:,:,1,:),2,10));
                title('Original patches.');
                subplot(5,1,2);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s2_p(:,:,1,:),'global'),2,10));
                title('Results from filter 1.');
                subplot(5,1,3);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s2_p(:,:,2,:),'global'),2,10));
                title('Results from filter 2.');
                subplot(5,1,4);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s2_p(:,:,3,:),'global'),2,10));
                title('Results from filter 3.');
                subplot(5,1,5);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s2_p(:,:,4,:),'global'),2,10));
                title('Results from filter 4.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    With MNIST test data.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = dataset.load_image_mnist('../test/mnist/t10k-images-idx3-ubyte','../test/mnist/t10k-labels-idx1-ubyte');
                      
            t = transforms.image.random_corr(s,4,5,5,@transforms.image.random_corr.sqr,4,log);
            s_p = t.code(s,log);
            
            assert(tc.tensor(s_p,4));
            assert(tc.same(size(s_p),[6 6 4 10000]));
            assert(tc.number(s_p));
            
            if exist('display','var') && (display)
                subplot(5,1,1);
                imshow(utils.format_as_tiles(s(:,:,1,1:1000:end),2,5));
                title('Original patches.');
                subplot(5,1,2);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p(:,:,1,1:1000:end),'global'),2,5));
                title('Results from filter 1.');
                subplot(5,1,3);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p(:,:,2,1:1000:end),'global'),2,5));
                title('Results from filter 2.');
                subplot(5,1,4);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p(:,:,3,1:1000:end),'global'),2,5));
                title('Results from filter 3.');
                subplot(5,1,5);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p(:,:,4,1:1000:end),'global'),2,5));
                title('Results from filter 4.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
