classdef deform < transform
    properties (GetAccess=public,SetAccess=immutable)
        scaling_max;
        rotation_max;
        field_smoothness_factor;
        field_intensity;
        do_deforming;
    end
    
    methods (Access=public)
        function [obj] = deform(train_sample_plain,scaling_max,rotation_max,field_smoothness_factor,field_intensity,logger)
            assert(tc.dataset_image(train_sample_plain));
            assert(size(train_sample_plain,3) == 1); % A BIT OF A HACK
            assert(size(train_sample_plain,1) == size(train_sample_plain,2)); % A BIT OF A HACK
            assert(tc.scalar(scaling_max));
            assert(tc.number(scaling_max));
            assert(scaling_max > 0);
            assert(scaling_max < 100);
            assert(tc.scalar(rotation_max));
            assert(tc.number(rotation_max));
            assert(rotation_max > 0);
            assert(rotation_max < 180);
            assert(tc.empty(field_smoothness_factor) || tc.scalar(field_smoothness_factor));
            assert(tc.empty(field_smoothness_factor) || tc.number(field_smoothness_factor));
            assert(tc.empty(field_smoothness_factor) || field_smoothness_factor > 0);
            assert(tc.empty(field_intensity) || tc.scalar(field_intensity));
            assert(tc.empty(field_intensity) || tc.number(field_intensity));
            assert(tc.empty(field_intensity) || field_intensity > 0);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            assert((tc.empty(field_smoothness_factor) && tc.empty(field_intensity)) || ...
                   (~tc.empty(field_smoothness_factor) && ~tc.empty(field_intensity)));
               
            if ~tc.empty(field_smoothness_factor)
                do_deforming_t = true;
            else
                do_deforming_t = false;
            end
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = input_geometry;
            
            obj = obj@transform(input_geometry,output_geometry,logger);
            obj.scaling_max = scaling_max;
            obj.rotation_max = rotation_max;
            obj.field_smoothness_factor = field_smoothness_factor;
            obj.field_intensity = field_intensity;
            obj.do_deforming = do_deforming_t;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain,logger)
            N = dataset.count(sample_plain);
            [~,dr,dc,~] = dataset.geometry(sample_plain);
            [col_i,row_i] = meshgrid(1:dr,1:dc);
            log_batch_size = ceil(N / 10);

            if obj.do_deforming
                smooth_kernel = fspecial('gaussian',dr,obj.field_smoothness_factor);
            end

            sample_coded = zeros(dr,dc,1,N);
            
            logger.beg_node('Building new images');
            
            for ii = 1:N
                if mod(ii - 1,log_batch_size) == 0
                    logger.message('Images %d to %d.',ii,min(ii + log_batch_size - 1,N));
                end

                scaling_factor_row = utils.rand_range(1 - obj.scaling_max/100,1 + obj.scaling_max/100);
                scaling_factor_col = utils.rand_range(1 - obj.scaling_max/100,1 + obj.scaling_max/100);
                rotation_angle = utils.rand_range(-obj.rotation_max,obj.rotation_max);
                
                original_instance = sample_plain(:,:,:,ii);
                
                [min_row,max_row,min_col,max_col] = transforms.image.digit.deform.get_bounds(original_instance);
                
                resized_row_count = min(dr,transforms.image.digit.deform.ceil_to_same_parity(dr,(max_row - min_row + 1) * scaling_factor_row));
                resized_col_count = min(dc,transforms.image.digit.deform.ceil_to_same_parity(dc,(max_col - min_col + 1) * scaling_factor_col));
                offset_row = (dr - resized_row_count) / 2 + 1;
                offset_col = (dc - resized_col_count) / 2 + 1;
                
                resized_core = imresize(original_instance(min_row:max_row,min_col:max_col),[resized_row_count resized_col_count]);
                resized_instance = zeros(dr,dc);
                resized_instance(offset_row:dr - offset_row + 1,offset_col:dc - offset_col + 1) = resized_core;
                
                rotated_instance = imrotate(resized_instance,rotation_angle,'bilinear','crop');
                
                if obj.do_deforming
                    field_row_1 = utils.rand_range(-0.05,0.05,dr,dc);
                    field_row_2 = conv2(field_row_1,smooth_kernel,'same');
                    field_row_3 = field_row_2 ./ max(max(field_row_2));
                    field_col_1 = utils.rand_range(-0.05,0.05,dr,dc);
                    field_col_2 = conv2(field_col_1,smooth_kernel,'same');
                    field_col_3 = field_col_2 ./ max(max(field_col_2));
                    
                    deformed_instance = interp2(col_i,row_i,rotated_instance,col_i + obj.field_intensity * field_col_3,row_i + obj.field_intensity * field_row_3,'bilinear',0);
                else
                    deformed_instance = rotated_instance;
                end
                
                sample_coded(:,:,:,ii) = deformed_instance;
            end
            
            logger.end_node();
        end
    end
    
    methods (Static,Access=private)
        function [min_row,max_row,min_col,max_col] = get_bounds(instance)
            [~,min_row] = find(instance',1,'first');
            [~,max_row] = find(instance',1,'last');
            [~,min_col] = find(instance,1,'first');
            [~,max_col] = find(instance,1,'last');
        end
        
        function [o] = ceil_to_same_parity(reference,num)
            if mod(reference,2) == 0
                o = 2 * ceil(num / 2);
            else
                o = 2 * ceil((num - 1) / 2) + 1;
            end
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.image.digit.deform".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    With field deformations.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = rand(16,16,1,100);
            
            t = transforms.image.digit.deform(s,10,15,5,2.4,log);
            
            assert(t.scaling_max == 10);
            assert(t.rotation_max == 15);
            assert(t.field_smoothness_factor == 5);
            assert(t.field_intensity == 2.4);
            assert(t.do_deforming == true);
            assert(tc.same(t.input_geometry,[16*16*1 16 16 1]));
            assert(tc.same(t.output_geometry,[16*16*1 16 16 1]));

            log.close();
            hnd.close();

            clearvars -except display;

            fprintf('    Without field deformations.\n');

            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = rand(16,16,1,100);

            t = transforms.image.digit.deform(s,10,15,[],[],log);

            assert(t.scaling_max == 10);
            assert(t.rotation_max == 15);
            assert(tc.same(t.field_smoothness_factor,[]));
            assert(tc.same(t.field_intensity,[]));
            assert(t.do_deforming == false);
            assert(tc.same(t.input_geometry,[16*16*1 16 16 1]));
            assert(tc.same(t.output_geometry,[16*16*1 16 16 1]));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s_1 = rand(16,16,1,16);
            s = zeros(24,24,1,16);
            s(5:20,5:20,:,:) = s_1;
            
            t = transforms.image.digit.deform(s,10,15,[],[],log);
            
            s_p = t.code(s,log);
            
            assert(tc.tensor(s_p,4));
            assert(tc.same(size(s_p),[24 24 1 16]));
            assert(tc.number(s_p));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                imshow(utils.format_as_tiles(s));
                subplot(1,2,2);
                imshow(utils.format_as_tiles(s_p));
                pause;
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
