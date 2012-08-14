classdef dc_offset < transform
    methods (Access=public)
        function [obj] = dc_offset(train_sample_plain)
            assert(check.dataset_record(train_sample_plain));
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = input_geometry;

            obj = obj@transform(input_geometry,output_geometry);
        end
    end
    
    methods (Access=protected)        
        function [sample_coded] = do_code(~,sample_plain)
            sample_coded = bsxfun(@minus,sample_plain,mean(sample_plain,1));
        end
    end

    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.dc_offset".\n');
            
            fprintf('  Proper construction.\n');
            
            s = [mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50)];
            
            t = transforms.record.dc_offset(s);
            
            assert(check.same(t.input_geometry,50));
            assert(check.same(t.output_geometry,50));

            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            s = [mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50)];
            
            t = transforms.record.dc_offset(s);
            s_p = t.code(s);
            
            assert(check.same(s_p,s - repmat(mean(s,1),50,1)));
            assert(check.same(mean(s_p,1),zeros(1,4)));
            
            if test_figure ~= -1
                figure(test_figure);
                for ii = 1:4
                    subplot(4,2,(ii - 1)*2 + 1);
                    plot(s(:,ii));
                    axis([1 50 -5 5]);
                    subplot(4,2,(ii - 1)*2 + 2);
                    plot(s_p(:,ii));
                    axis([1 50 -5 5]);
                end
                pause(5);
            end
            
            clearvars -except test_figure;
        end
    end
end
