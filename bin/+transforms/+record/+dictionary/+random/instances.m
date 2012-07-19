classdef instances < transforms.record.dictionary
    methods (Access=public)
        function [obj] = instances(train_sample_plain,word_count,coding_method,coding_params,do_polarity_split,logger)
            assert(check.dataset_record(train_sample_plain));
            assert(check.scalar(word_count));
            assert(check.natural(word_count));
            assert(word_count > 0);
            assert(transforms.record.dictionary.coding_setup_ok(word_count,coding_method,coding_params));
            assert(check.scalar(do_polarity_split));
            assert(check.logical(do_polarity_split));
            assert(check.scalar(logger));
            assert(check.logging_logger(logger));
            assert(logger.active);
            assert(dataset.count(train_sample_plain) >= word_count);
            
            N = dataset.count(train_sample_plain);
            dict = dataset.subsample(train_sample_plain,randi(N,1,word_count))';
            
            obj = obj@transforms.record.dictionary(train_sample_plain,dict,coding_method,coding_params,do_polarity_split,logger);
        end
    end
    
    methods (Static,Access=public)
        function test(test_figure)
            fprintf('Testing "transforms.record.dictionary.random.instances".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.Experiment);
            logg = logging.logger({hnd});
            s = utils.testing.three_component_cloud();
            
            t = transforms.record.dictionary.random.instances(s,3,'Corr',[],false,logg);
            
            assert(check.matrix(t.dict));
            assert(check.same(size(t.dict),[3 2]));
            assert(check.number(t.dict));
            assert(check.checkf(@(ii)check.same(norm(t.dict(ii,:)),1),1:3));
            assert(check.matrix(t.dict_transp));
            assert(check.same(size(t.dict_transp),[2 3]));
            assert(check.number(t.dict_transp));
            assert(check.checkf(@(ii)check.same(norm(t.dict_transp(:,ii)),1),1:3));
            assert(check.same(t.dict',t.dict_transp));
            assert(t.word_count == 3);
            assert(check.same(t.coding_fn,@transforms.record.dictionary.correlation));
            assert(check.same(t.coding_params_cell,{}));
            assert(check.same(t.coding_method,'Corr'));
            assert(check.same(t.coding_params,[]));
            assert(check.same(t.input_geometry,2));
            assert(check.same(t.output_geometry,3));

            s_p = t.code(s,logg);
            s_r = t.dict_transp * s_p;

            if test_figure ~= -1
                figure(test_figure);
                clf(gcf());
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                hold off;
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
            end

            logg.close();
            hnd.close();
            
            clearvars -except test_figure;
        end
    end
end
