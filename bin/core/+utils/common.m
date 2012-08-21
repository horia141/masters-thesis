classdef common
    methods (Static,Access=public)
        function [o] = rand_range(a,b,varargin)
            assert(check.scalar(a));
            assert(check.number(a));
            assert(check.scalar(b));
            assert(check.scalar(b));
            assert(check.empty(varargin) || check.vector(varargin));
            assert(check.empty(varargin) || check.checkf(@check.scalar,varargin));
            assert(check.empty(varargin) || check.checkf(@check.number,varargin));
            assert(a < b);
            
            o = (b - a) * rand(varargin{:}) + a;
        end
        
        function [o_v] = force_row(v)
            assert(check.vector(v));
            
            if size(v,2) == length(v)
                o_v = v;
            else
                o_v = v';
            end
        end
        
        function [o_v] = force_col(v)
            assert(check.vector(v));
            
            if size(v,1) == length(v)
                o_v = v;
            else
                o_v = v';
            end
        end
        
        function [o_v] = force_same(v,other_v)
            assert(check.vector(v));
            assert(check.vector(other_v));
            
            if size(other_v,2) == length(other_v)
                o_v = utils.common.force_row(v);
            else
                o_v = utils.common.force_col(v);
            end
        end
        
        function [new_images] = clamp_images_to_unit(images)
            assert(check.dataset_image(images));
            
            new_images = max(min(images,1),0);
        end
        
        function [new_images] = remap_images_to_unit(images)
            assert(check.dataset_image(images));
            
            im_min = min(images(:));
            im_max = max(images(:));
                
            new_images = (images - im_min) / (im_max - im_min);
        end
        
        function [s] = value_to_string(i)
            assert(check.scalar(i));
            assert(check.value(i));
            
            if check.logical(i)
                if i
                    s = 'true';
                else
                    s = 'false';
                end
            elseif check.number(i)
                if check.integer(i)
                    s = sprintf('%d',i);
                else
                    s = sprintf('%f',i);
                end
            elseif check.string(i)
                s = i;
            elseif check.function_h(i)
                s = func2str(i);
            else
                assert(false);
            end
        end
        
        function [s] = matrix_to_string(mat,format)
            assert(check.matrix(mat));
            assert(check.number(mat));
            assert(~exist('format','var') || check.scalar(format));
            assert(~exist('format','var') || check.string(format));

            if exist('format','var')
                format_t = format;
            else
                format_t = '%f ';
            end
            
            contents = cell(size(mat));
            
            for ii = 1:size(mat,1)
                for jj = 1:size(mat,2)
                    contents{ii,jj} = sprintf(format_t,mat(ii,jj));
                end
            end
            
            max_size = max(cellfun(@length,contents(:)));
            size_format = sprintf('%%%ds',max_size);
                
            s = '';
            
            for ii = 1:size(mat,1)
                for jj = 1:size(mat,2)
                    s = sprintf('%s%s',s,sprintf(size_format,contents{ii,jj}));
                end
                
                s = sprintf('%s\n',s);
            end
        end
        
        function [sched] = schedule(initial,final,T_max)
            assert(check.scalar(initial));
            assert(check.number(initial));
            assert(initial > 0);
            assert(check.scalar(final));
            assert(check.number(final));
            assert(final > 0);
            assert(check.scalar(T_max));
            assert(check.natural(T_max));
            assert(T_max >= 1);
            assert(final <= initial);

            if T_max ~= 1
                sched = initial * (final / initial) .^ ((0:(T_max - 1)) / (T_max - 1));
            else
                sched = initial;
            end
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "utils.common".\n');
            
            fprintf('  Function "rand_range".\n');
            
            r1 = utils.common.rand_range(0,1);
            r2 = utils.common.rand_range(-1,1);
            r3 = utils.common.rand_range(0,10,1,10);
            r4 = utils.common.rand_range(-3,3,4,4,5);
            
            assert(check.scalar(r1));
            assert(check.unitreal(r1));
            assert(check.scalar(r2));
            assert(check.unitreal(abs(r2)));
            assert(check.vector(r3));
            assert(length(r3) == 10);
            assert(check.checkv(r3 >= 0));
            assert(check.checkv(r3 <= 10));
            assert(check.tensor(r4,3));
            assert(size(r4,1) == 4);
            assert(size(r4,2) == 4);
            assert(size(r4,3) == 5);
            assert(check.checkv(r4 >= -3));
            assert(check.checkv(r4 <= 3));
            
            clearvars -except test_figure;
            
            fprintf('  Function "force_row".\n');
            
            assert(check.same(utils.common.force_row(1),1));
            assert(check.same(utils.common.force_row([1 2 3]),[1 2 3]));
            assert(check.same(utils.common.force_row([1;2;3]),[1 2 3]));
            assert(check.same(utils.common.force_row(zeros(1,45)),zeros(1,45)));
            assert(check.same(utils.common.force_row(ones(41,1)),ones(1,41)));
            assert(check.same(utils.common.force_row({1 2 3}),{1 2 3}));
            assert(check.same(utils.common.force_row({1;2;3}),{1 2 3}));
            assert(check.same(utils.common.force_row({'hello';'world'}),{'hello' 'world'}));
            
            clearvars -except test_figure;
            
            fprintf('  Function "force_col".\n');

            assert(check.same(utils.common.force_col(1),1));
            assert(check.same(utils.common.force_col([1 2 3]),[1;2;3]));
            assert(check.same(utils.common.force_col([1;2;3]),[1;2;3]));
            assert(check.same(utils.common.force_col(zeros(1,45)),zeros(45,1)));
            assert(check.same(utils.common.force_col(ones(41,1)),ones(41,1)));
            assert(check.same(utils.common.force_col({1 2 3}),{1;2;3}));
            assert(check.same(utils.common.force_col({1;2;3}),{1;2;3}));
            assert(check.same(utils.common.force_col({'hello' 'world'}),{'hello';'world'}));
            
            clearvars -except test_figure;
            
            fprintf('  Function "force_same".\n');
            
            assert(check.same(utils.common.force_same(1,true),1));
            assert(check.same(utils.common.force_same([1 2 3],[true true false]),[1 2 3]));
            assert(check.same(utils.common.force_same([1 2 3],[true;true;false]),[1;2;3]));
            assert(check.same(utils.common.force_same([1;2],[true true false]),[1 2]));
            assert(check.same(utils.common.force_same([1;2],[true;true;false]),[1;2]));
            
            clearvars -except test_figure;
            
            fprintf('  Function "clamp_images_to_unit".\n');
            
            fprintf('    With small excursions below 0 and above 1.\n');
            
            t = dataset.load('../../test/scenes_small.mat') + 0.2;
            tp = utils.common.clamp_images_to_unit(t);
            
            assert(dataset.geom_compatible(dataset.geometry(t),dataset.geometry(tp)));
            assert(min(tp(:)) >= 0);
            assert(max(tp(:)) <= 1);
            assert(check.unitreal(tp));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                utils.display.as_tiles(t);
                title('Problem images.');
                subplot(1,2,2);
                utils.display.as_tiles(tp);
                title('Clamped images.');
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('    With large excursions below 0 and above 1.\n');
            
            t = 4 * dataset.load('../../test/scenes_small.mat') - 2;
            tp = utils.common.clamp_images_to_unit(t);
            
            assert(dataset.geom_compatible(dataset.geometry(t),dataset.geometry(tp)));
            assert(min(tp(:)) >= 0);
            assert(max(tp(:)) <= 1);
            assert(check.unitreal(tp));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                utils.display.as_tiles(t);
                title('Problem images.');
                subplot(1,2,2);
                utils.display.as_tiles(tp);
                title('Clamped images.');
                pause(5);
            end
            
            clearvars -except test_figure;

            fprintf('  Function "remap_images_to_unit".\n');
            
            t = 4 * dataset.load('../../test/scenes_small.mat') - 2;
            t(:,:,:,4:5) = 4 * t(:,:,:,4:5);
            tp = utils.common.remap_images_to_unit(t);
            
            assert(dataset.geom_compatible(dataset.geometry(t),dataset.geometry(tp)));
            assert(check.same(min(tp(:)),0));
            assert(check.same(max(tp(:)),1));
            assert(check.unitreal(tp));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                utils.display.as_tiles(t);
                title('Problem images.');
                subplot(1,2,2);
                utils.display.as_tiles(tp);
                title('Remaped images.');
                pause(5);
            end
            
            clearvars -except test_figure;

            fprintf('  Function "value_to_string".\n');
            
            assert(check.same(utils.common.value_to_string(true),'true'));
            assert(check.same(utils.common.value_to_string(false),'false'));
            assert(check.same(utils.common.value_to_string(7.3),'7.300000'));
            assert(check.same(utils.common.value_to_string(-5),'-5'));
            assert(check.same(utils.common.value_to_string(7),'7'));
            assert(check.same(utils.common.value_to_string('hello'),'hello'));
            assert(check.same(utils.common.value_to_string(@(c)c),'@(c)c'));
            assert(check.same(utils.common.value_to_string(@utils.common.value_to_string),'utils.common.value_to_string'));
            
            clearvars -except test_figure;
            
            fprintf('  Function "matrix_to_string".\n');
            
            assert(check.same(utils.common.matrix_to_string([1 0; 0 1]),sprintf('1.000000 0.000000 \n0.000000 1.000000 \n')));
            assert(check.same(utils.common.matrix_to_string([1 2 3; 4 5 6; 7 8 9]),sprintf('1.000000 2.000000 3.000000 \n4.000000 5.000000 6.000000 \n7.000000 8.000000 9.000000 \n')));
            assert(check.same(utils.common.matrix_to_string([1 0; 0 1],'%d '),sprintf('1 0 \n0 1 \n')));
            assert(check.same(utils.common.matrix_to_string([1.2 3.2; 4.4 1.3],' %.2f '),sprintf(' 1.20  3.20 \n 4.40  1.30 \n')));
            
            clearvars -except test_figure;
            
            fprintf('  Function "schedule".\n');
            
            assert(check.same(utils.common.schedule(1,0.5,2),[1 0.5]));
            assert(check.same(utils.common.schedule(1,0.1,3),[1 0.3162 0.1],1e-2));
            assert(check.same(utils.common.schedule(1,0.1,1),1));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(2,2,1);
                plot(1:10,utils.common.schedule(1,0.1,10));
                axis([1 10 0 1]);
                subplot(2,2,2);
                plot(1:20,utils.common.schedule(2,0.5,20));
                axis([1 20 0 2]);
                subplot(2,2,3);
                plot(1:20,utils.common.schedule(10,1e-3,20));
                axis([1 20 0 10]);
                subplot(2,2,4);
                plot(1:100,utils.common.schedule(1,0.5,100));
                axis([1 100 0 1]);
                pause(5);
            end
            
            clearvars -except test_figure;
        end
    end
end
