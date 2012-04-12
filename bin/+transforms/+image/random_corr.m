classdef random_corr < transform
    properties (GetAccess=public,SetAccess=immutable)
        filters;
        filters_count;
        filter_row_count;
        filter_col_count;
        reduce_fn;
        reduce_spread;
    end
    
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = random_corr(train_image_plain,filters_count,filter_row_count,filter_col_count,reduce_fn,reduce_spread,logger)
            assert(tc.scalar(train_image_plain));
            assert(tc.datasets_image(train_image_plain));
            assert(train_image_plain.samples_count >= 1);
            assert(train_image_plain.layers_count == 1);
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
            assert(mod(train_image_plain.row_count - filter_row_count + 1,reduce_spread) == 0);
            assert(mod(train_image_plain.col_count - filter_col_count + 1,reduce_spread) == 0);
            
            logger.message('Filter bank size: %dx%d',(train_image_plain.row_count - filter_row_count + 1) / reduce_spread,...
                                                     (train_image_plain.col_count - filter_col_count + 1) / reduce_spread); 
            logger.message('Building filter bank.');
            
            
            filters_t = 0.1 * rand(filter_row_count,filter_col_count,filters_count) - 0.05;
            
            obj = obj@transform(logger);
            obj.filters = filters_t;
            obj.filters_count = filters_count;
            obj.filter_row_count = filter_row_count;
            obj.filter_col_count = filter_col_count;
            obj.reduce_fn = reduce_fn;
            obj.reduce_spread = reduce_spread;
            
            logger.beg_node('Extracting plain/coded samples');
            
            obj.one_sample_plain = train_image_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain,logger);
            
            logger.end_node();
        end
    end
    
    methods (Access=protected)
        function [image_coded] = do_code(obj,image_plain,logger)
            image_coded_row_count = (image_plain.row_count - obj.filter_row_count + 1) / obj.reduce_spread;
            image_coded_col_count = (image_plain.col_count - obj.filter_col_count + 1) / obj.reduce_spread;
            
            images_coded = zeros(image_coded_row_count,image_coded_col_count,obj.filters_count,image_plain.samples_count);
            
            logger.beg_node('Applying filters to each image');
            
            for filter = 1:obj.filters_count
                logger.beg_node('Filter #%d',filter);
                
                logger.message('Convolving with filter kernel.');

                conv_images = arrayfun(@(ii)conv2(image_plain.images(:,:,1,ii),obj.filters(:,:,filter),'valid'),...
                                       1:image_plain.samples_count,'UniformOutput',false);
                conv_images = cat(3,conv_images{:});
                
                logger.message('Reducing resulting images.');
            
                for ii = 1:image_coded_row_count
                    for jj = 1:image_coded_col_count
                        images_coded(ii,jj,filter,:) = obj.reduce_fn(conv_images(((ii-1)*obj.reduce_spread + 1):(ii*obj.reduce_spread),...
                                                                                 ((jj-1)*obj.reduce_spread + 1):(jj*obj.reduce_spread),:));
                    end
                end
                
                logger.end_node();
            end
            
            logger.end_node();
            
            logger.message('Building dataset.');

            image_coded = datasets.image(image_plain.classes,images_coded,image_plain.labels_idx);
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
            s1 = datasets.image.load_from_dir('../data/test/scenes_small');
            t1 = transforms.image.patch_extract(s1,20,32,32,0,log);
            s2 = t1.code(s1,log);
            
            t2 = transforms.image.random_corr(s2,4,5,5,@transforms.image.random_corr.sqr,4,log);
            
            assert(tc.check(size(t2.filters) == [5 5 4]));
            assert(tc.tensor(t2.filters,3) && tc.check(t2.filters >= -0.05 & t2.filters <= 0.05));
            assert(t2.filters_count == 4);
            assert(t2.filter_row_count == 5);
            assert(t2.filter_col_count == 5);
            assert(strcmp(func2str(t2.reduce_fn),'transforms.image.random_corr.sqr'));
            assert(t2.reduce_spread == 4);
            assert(length(t2.one_sample_plain.classes) == 1);
            assert(strcmp(t2.one_sample_plain.classes{1},'none'));
            assert(t2.one_sample_plain.classes_count == 1);
            assert(tc.check(t2.one_sample_plain.samples == s2.samples(1,:)));
            assert(tc.check(t2.one_sample_plain.labels_idx == s2.labels_idx(1)));
            assert(t2.one_sample_plain.samples_count == 1);
            assert(t2.one_sample_plain.features_count == 32*32);
            assert(tc.check(t2.one_sample_plain.images == s2.images(:,:,:,1)));
            assert(t2.one_sample_plain.layers_count == 1);
            assert(t2.one_sample_plain.row_count == 32);
            assert(t2.one_sample_plain.col_count == 32);
            assert(t2.one_sample_plain.compatible(s2));
            assert(length(t2.one_sample_coded.classes) == 1);
            assert(strcmp(t2.one_sample_coded.classes{1},'none'));
            assert(t2.one_sample_coded.classes_count == 1);
            assert(tc.check(size(t2.one_sample_coded.samples) == [1 4*7*7]));
            assert(tc.matrix(t2.one_sample_coded.samples) && tc.unitreal(t2.one_sample_coded.samples));
            assert(tc.check(t2.one_sample_coded.labels_idx == s2.labels_idx(1)));
            assert(t2.one_sample_coded.samples_count == 1);
            assert(t2.one_sample_coded.features_count == 4*7*7);
            assert(tc.check(size(t2.one_sample_coded.images) == [7 7 4]));
            assert(tc.tensor(t2.one_sample_coded.images,4) && tc.unitreal(t2.one_sample_coded.images));
            assert(t2.one_sample_coded.layers_count == 4);
            assert(t2.one_sample_coded.row_count == 7);
            assert(t2.one_sample_coded.col_count == 7);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Extracting patches:\n',...
                                                          '    Patches 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Extracting patches:\n',...
                                                          '  Patches 1 to 2.\n',...
                                                          '  Patches 3 to 4.\n',...
                                                          '  Patches 5 to 6.\n',...
                                                          '  Patches 7 to 8.\n',...
                                                          '  Patches 9 to 10.\n',...
                                                          '  Patches 11 to 12.\n',...
                                                          '  Patches 13 to 14.\n',...
                                                          '  Patches 15 to 16.\n',...
                                                          '  Patches 17 to 18.\n',...
                                                          '  Patches 19 to 20.\n',...
                                                          'Building dataset.\n',...
                                                          'Filter bank size: 7x7\n',...
                                                          'Building filter bank.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Applying filters to each image:\n',...
                                                          '    Filter #1:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '    Filter #2:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '    Filter #3:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '    Filter #4:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '  Building dataset.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    With random patches from "scenes_small".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s1 = datasets.image.load_from_dir('../data/test/scenes_small');
            t1 = transforms.image.patch_extract(s1,20,32,32,0,log);
            s2 = t1.code(s1,log);
            
            t2 = transforms.image.random_corr(s2,4,5,5,@transforms.image.random_corr.sqr,4,log);
            s2_p = t2.code(s2,log);
            
            assert(length(s2_p.classes) == 1);
            assert(strcmp(s2_p.classes{1},'none'));
            assert(s2_p.classes_count == 1);
            assert(tc.check(size(s2_p.samples) == [20 4*7*7]));
            assert(tc.matrix(s2_p.samples) && tc.number(s2_p.samples));
            assert(tc.check(s2_p.labels_idx == ones(20,1)));
            assert(s2_p.features_count == 4*7*7);
            assert(tc.check(size(s2_p.images) == [7 7 4 20]));
            assert(tc.tensor(s2_p.images,4) && tc.number(s2_p.images));
            assert(s2_p.layers_count == 4);
            assert(s2_p.row_count == 7);
            assert(s2_p.col_count == 7);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Extracting patches:\n',...
                                                          '    Patches 1 to 1.\n',...
                                                          '  Building dataset.\n',...
                                                          'Extracting patches:\n',...
                                                          '  Patches 1 to 2.\n',...
                                                          '  Patches 3 to 4.\n',...
                                                          '  Patches 5 to 6.\n',...
                                                          '  Patches 7 to 8.\n',...
                                                          '  Patches 9 to 10.\n',...
                                                          '  Patches 11 to 12.\n',...
                                                          '  Patches 13 to 14.\n',...
                                                          '  Patches 15 to 16.\n',...
                                                          '  Patches 17 to 18.\n',...
                                                          '  Patches 19 to 20.\n',...
                                                          'Building dataset.\n',...
                                                          'Filter bank size: 7x7\n',...
                                                          'Building filter bank.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Applying filters to each image:\n',...
                                                          '    Filter #1:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '    Filter #2:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '    Filter #3:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '    Filter #4:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '  Building dataset.\n',...
                                                          'Applying filters to each image:\n',...
                                                          '  Filter #1:\n',...
                                                          '    Convolving with filter kernel.\n',...
                                                          '    Reducing resulting images.\n',...
                                                          '  Filter #2:\n',...
                                                          '    Convolving with filter kernel.\n',...
                                                          '    Reducing resulting images.\n',...
                                                          '  Filter #3:\n',...
                                                          '    Convolving with filter kernel.\n',...
                                                          '    Reducing resulting images.\n',...
                                                          '  Filter #4:\n',...
                                                          '    Convolving with filter kernel.\n',...
                                                          '    Reducing resulting images.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display)
                subplot(5,1,1);
                imshow(utils.format_as_tiles(s2.images(:,:,1,:),2,10));
                title('Original patches.');
                subplot(5,1,2);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s2_p.images(:,:,1,:),'global'),2,10));
                title('Results from filter 1.');
                subplot(5,1,3);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s2_p.images(:,:,2,:),'global'),2,10));
                title('Results from filter 2.');
                subplot(5,1,4);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s2_p.images(:,:,3,:),'global'),2,10));
                title('Results from filter 3.');
                subplot(5,1,5);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s2_p.images(:,:,4,:),'global'),2,10));
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
            s = datasets.image.load_mnist('../data/test/mnist/t10k-images-idx3-ubyte','../data/test/mnist/t10k-labels-idx1-ubyte');
                      
            t = transforms.image.random_corr(s,4,5,5,@transforms.image.random_corr.sqr,4,log);
            s_p = t.code(s,log);
            
            assert(length(s_p.classes) == 10);
            assert(strcmp(s_p.classes{1},'0'));
            assert(strcmp(s_p.classes{2},'1'));
            assert(strcmp(s_p.classes{3},'2'));
            assert(strcmp(s_p.classes{4},'3'));
            assert(strcmp(s_p.classes{5},'4'));
            assert(strcmp(s_p.classes{6},'5'));
            assert(strcmp(s_p.classes{7},'6'));
            assert(strcmp(s_p.classes{8},'7'));
            assert(strcmp(s_p.classes{9},'8'));
            assert(strcmp(s_p.classes{10},'9'));
            assert(s_p.classes_count == 10);
            assert(tc.check(size(s_p.samples) == [10000 4*6*6]));
            assert(tc.matrix(s_p.samples) && tc.number(s_p.samples));
            assert(tc.check(s_p.labels_idx == s.labels_idx));
            assert(s_p.features_count == 4*6*6);
            assert(tc.check(size(s_p.images) == [6 6 4 10000]));
            assert(tc.tensor(s_p.images,4) && tc.number(s_p.images));
            assert(s_p.layers_count == 4);
            assert(s_p.row_count == 6);
            assert(s_p.col_count == 6);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Filter bank size: 6x6\n',...
                                                          'Building filter bank.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Applying filters to each image:\n',...
                                                          '    Filter #1:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '    Filter #2:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '    Filter #3:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '    Filter #4:\n',...
                                                          '      Convolving with filter kernel.\n',...
                                                          '      Reducing resulting images.\n',...
                                                          '  Building dataset.\n',...
                                                          'Applying filters to each image:\n',...
                                                          '  Filter #1:\n',...
                                                          '    Convolving with filter kernel.\n',...
                                                          '    Reducing resulting images.\n',...
                                                          '  Filter #2:\n',...
                                                          '    Convolving with filter kernel.\n',...
                                                          '    Reducing resulting images.\n',...
                                                          '  Filter #3:\n',...
                                                          '    Convolving with filter kernel.\n',...
                                                          '    Reducing resulting images.\n',...
                                                          '  Filter #4:\n',...
                                                          '    Convolving with filter kernel.\n',...
                                                          '    Reducing resulting images.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display)
                subplot(5,1,1);
                imshow(utils.format_as_tiles(s.images(:,:,1,1:1000:end),2,5));
                title('Original patches.');
                subplot(5,1,2);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p.images(:,:,1,1:1000:end),'global'),2,5));
                title('Results from filter 1.');
                subplot(5,1,3);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p.images(:,:,2,1:1000:end),'global'),2,5));
                title('Results from filter 2.');
                subplot(5,1,4);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p.images(:,:,3,1:1000:end),'global'),2,5));
                title('Results from filter 3.');
                subplot(5,1,5);
                imshow(utils.format_as_tiles(utils.remap_images_to_unit(s_p.images(:,:,4,1:1000:end),'global'),2,5));
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
