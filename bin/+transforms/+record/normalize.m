classdef normalize < transform
    properties (GetAccess=public,SetAccess=immutable)
        kept_mean;
    end
    
    methods (Access=public)
        function [obj] = normalize(train_sample_plain)
            assert(check.dataset_record(train_sample_plain));
            
            kept_mean_t = mean(train_sample_plain,2);
            
            input_geometry = dataset.geometry(train_sample_plain);
            output_geometry = input_geometry;
            
            obj = obj@transform(input_geometry,output_geometry);
            obj.kept_mean = kept_mean_t;
        end
    end
    
    methods (Access=protected)
        function [sample_coded] = do_code(obj,sample_plain)
            sample_coded_t1 = bsxfun(@minus,sample_plain,obj.kept_mean);
            sample_coded = bsxfun(@rdivide,sample_coded_t1,sqrt(sum(sample_coded_t1 .^ 2,1)));
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.normalize".\n');
            
            fprintf('  Proper construction.\n');
            
            s = mvnrnd([3 3],[1 0; 0 0.3],10000)';
            
            t = transforms.record.normalize(s);
    
            assert(check.same(t.kept_mean,[3;3],0.1));
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,2));
            
            clearvars -except test_figure;
            
            fprintf('  Function "code".\n');
            
            s = mvnrnd([3 3],[1 0; 0 0.3],10000)';
            
            t = transforms.record.normalize(s);
            s_p = t.code(s);
            
            % Use "mean(s,2)" instead of [3;3] because of precision problems.
            assert(check.same(s_p,(s - repmat(mean(s,2),1,10000)) ./ repmat(sqrt(sum((s - repmat(mean(s,2),1,10000)) .^ 2,1)),2,1),0.1));
            assert(check.same(mean(s_p,2),[0;0],0.1));
            assert(check.same(sqrt(sum(s_p .^ 2,1)),ones(1,10000),0.1));
            
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
                title('Normalized sample.');
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('  With a more complex sample.\n');
            
            s = [mvnrnd([3 3],[0.1 0; 0 0.03],10000);
                 mvnrnd([1 1],[0.1 0; 0 0.03],10000)]';

            t = transforms.record.normalize(s);
            s_p = t.code(s);
            
            % Use "mean(s,2)" instead of [3;3] because of precision problems.
            assert(check.same(s_p,(s - repmat(mean(s,2),1,20000)) ./ repmat(sqrt(sum((s - repmat(mean(s,2),1,20000)) .^ 2,1)),2,1),0.1));
            assert(check.same(mean(s_p,2),[0;0],0.1));
            assert(check.same(sqrt(sum(s_p .^ 2,1)),ones(1,20000),0.1));
            
            if test_figure ~= -1
                figure(test_figure);
                subplot(1,2,1);
                hold('on');
                scatter(s(1,1:10000),s(2,1:10000),'r','o');
                scatter(s(1,10001:20000),s(2,10001:20000),'b','o');
                hold('off');
                axis([-4 6 -4 6]);
                axis('square');
                title('Original sample.');
                subplot(1,2,2);
                hold('on');
                scatter(s_p(1,1:10000),s_p(2,1:10000),'r','x');
                scatter(s_p(1,10001:20000),s_p(2,10001:20000),'b','x');
                hold('off');
                axis([-4 6 -4 6]);
                axis('square');
                title('Normalized sample.');
                pause(5);
            end
            
            clearvars -except test_figure;
            
            fprintf('  With more complex instances.\n');
            
            s = [mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50),...
                 mvnrnd(randi(5) - 3,2,50)];
            
            t = transforms.record.normalize(s);
            s_p = t.code(s);
            
            % Use "mean(s,2)" instead of [3;3] because of precision problems.
            assert(check.same(s_p,(s - repmat(mean(s,2),1,4)) ./ repmat(sqrt(sum((s - repmat(mean(s,2),1,4)) .^ 2,1)),50,1),0.1));
            assert(check.same(mean(s_p,2),zeros(50,1),0.1));
            assert(check.same(sqrt(sum(s_p .^ 2,1)),ones(1,4),0.1));
            
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
