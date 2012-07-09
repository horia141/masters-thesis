classdef resize < transform
    properties (GetAccess=public,SetAccess=immutable)
        new_row_count;
        new_col_count;
    end
    
    methods (Access=public)
        function [obj] = resize(train_sample_plain,new_row_count,new_col_count,logger)
            assert(check.dataset_image(train_sample_plain));
            assert(check.scalar(new_row_count));
            assert(check.natural(new_row_count));
            assert(new_row_count >= 1);
            assert(check.scalar(new_col_count));
            assert(check.natural(new_col_count));
            assert(new_col_count >= 1);
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
            assert(logger.active);
            
            [d,dr,dc,dl] = dataset.geometry(train_sample_plain);
            
            input_geometry = [d dr dc dl];
            output_geometry = [new_row_count * new_col_count * dl new_row_count new_col_count dl];
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.new_row_count = new_row_count;
            obj.new_col_count = new_col_count;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            [~,dr,dc,dl] = dataset.geometry(sample_plain);            

            if mod(dr,obj.new_row_count) == 0 && mod(dc,obj.new_col_count) == 0
                logger.message('Building resized images (fast because new size is a sub-multiple of old size).');

                row_step = dr / obj.new_row_count;
                col_step = dc / obj.new_col_count;
                
                sample_coded = sample_plain(1:row_step:end,1:col_step:end,:,:);
            else
                N = dataset.count(sample_plain);

                log_batch_size = ceil(N / 10);
                sample_coded = zeros(obj.new_row_count,obj.new_col_count,dl,N);

                logger.beg_node('Building resized images');

                for ii = 1:N
                    if mod(ii - 1,log_batch_size) == 0
                        logger.message('Processing images %d to %d.',ii,min(ii + log_batch_size - 1,N));
                    end

                    sample_coded(:,:,:,ii) = imresize(sample_plain(:,:,:,ii),[obj.new_row_count obj.new_col_count]);
                end

                logger.end_node();
            end
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.image.resize".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            
            t = transforms.image.resize(s,20,20,logg);
            
            assert(t.new_row_count == 20);
            assert(t.new_col_count == 20);
            assert(check.same(t.input_geometry,[192*256*3 192 256 3]));
            assert(check.same(t.output_geometry,[20*20*3 20 20 3]));
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            
            t = transforms.image.resize(s,100,100,logg);
            s_p = t.code(s,logg);
            
            assert(check.checkf(@(ii)check.same(s_p(:,:,:,ii),imresize(s(:,:,:,ii),[100 100])),1:7));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                utils.display.as_tiles(s);
                subplot(1,2,2);
                utils.display.as_tiles(s_p);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
            
            fprintf('    With sub-multiple size.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.scenes_small();
            
            t = transforms.image.resize(s,96,128,logg);
            s_p = t.code(s,logg);

            assert(check.checkf(@(ii)check.same(s_p(:,:,:,ii),s(1:2:end,1:2:end,:,ii)),1:7));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                utils.display.as_tiles(s);
                subplot(1,2,2);
                utils.display.as_tiles(s_p);
                pause(5);
            end
            
            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
        end
    end
end
