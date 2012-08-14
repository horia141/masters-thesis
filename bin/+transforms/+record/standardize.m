classdef standardize < transform
    properties (GetAccess=public,SetAccess=immutable)
        kept_mean;
        kept_deviation;
    end
    
    methods (Access=public)
        function [obj] = standardize(train_sample_plain)
            assert(check.dataset_record(train_sample_plain));
            
            kept_mean_t = mean(train_sample_plain,2);
            kept_deviation_t = std(train_sample_plain,0,2);
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = input_geometry;
            
            obj = obj@transform(input_geometry,output_geometry);
            obj.kept_mean = kept_mean_t;
            obj.kept_deviation = kept_deviation_t;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain)
            sample_coded_t1 = bsxfun(@minus,sample_plain,obj.kept_mean);
            sample_coded = bsxfun(@rdivide,sample_coded_t1,obj.kept_deviation);
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.standardize".\n');
            
            fprintf('  Proper construction.\n');
            
            s = mvnrnd([3 3],[1 0; 0 0.3],10000)';
            
            t = transforms.record.standardize(s);
            
            assert(check.same(t.kept_mean,[3;3],0.1));
            assert(check.same(t.kept_deviation,[1;sqrt(0.3)],0.1));
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,2));
            
            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            s = mvnrnd([3 3],[1 0; 0 0.3],10000)';
            
            t = transforms.record.standardize(s);
            s_p = t.code(s);
            
            assert(check.same(s_p,(s - repmat([3;3],1,10000)) ./ repmat([1;sqrt(0.3)],1,10000),0.1));
            assert(check.same(mean(s_p,2),[0;0],0.1));
            assert(check.same(std(s_p,0,2),[1;1],0.1));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original sample.');
                subplot(1,2,2);
                scatter(s_p(1,:),s_p(2,:),'x');
                axis([-4 6 -4 6]);
                axis('square');
                title('Standardized sample.');
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('  With non-independent features.\n');
            
            s = mvnrnd([3 3],[1 0.4; 0.4 0.3],10000)';
            
            t = transforms.record.standardize(s);
            s_p = t.code(s);
            
            assert(check.same(s_p,(s - repmat([3;3],1,10000)) ./ repmat([1;sqrt(0.3)],1,10000),0.1));
            assert(check.same(mean(s_p,2),[0;0],0.1));
            assert(check.same(std(s_p,0,2),[1;1],0.1));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                scatter(s(1,:),s(2,:),'o');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original sample.');
                subplot(1,2,2);
                scatter(s_p(1,:),s_p(2,:),'x');
                axis([-4 6 -4 6]);
                axis('square');
                title('Standardized sample.');
                pause(5);
            end
            
            clearvars -except test_figure;
        end
    end
end
