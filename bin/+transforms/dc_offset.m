classdef dc_offset < transform
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = dc_offset(train_dataset_plain,logger)
            assert(tc.scalar(train_dataset_plain));
            assert(tc.dataset(train_dataset_plain));
            assert(train_dataset_plain.samples_count >= 1);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            obj = obj@transform(logger);
            
            logger.beg_node('Extracting plain/coded samples');
            
            obj.one_sample_plain = train_dataset_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain,logger);
            
            logger.end_node();
        end
    end
    
    methods (Access=protected)        
        function [dataset_coded] = do_code(~,dataset_plain,logger)
            logger.message('Substracting DC component from each sample.');

            samples_coded = bsxfun(@minus,dataset_plain.samples,mean(dataset_plain.samples,2));

            logger.message('Building dataset.');
            
            dataset_coded = dataset(dataset_plain.classes,samples_coded,dataset_plain.labels_idx);
        end
    end

    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.dc_offset".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)'];
            c = ones(4,1);
            s = dataset({'none'},A,c);
            
            t = transforms.dc_offset(s,log);
            
            assert(length(t.one_sample_plain.classes) == 1);
            assert(strcmp(t.one_sample_plain.classes{1},'none'));
            assert(t.one_sample_plain.classes_count == 1);
            assert(tc.check(t.one_sample_plain.samples == A(1,:)));
            assert(tc.check(t.one_sample_plain.labels_idx == c(1)));
            assert(t.one_sample_plain.samples_count == 1);
            assert(t.one_sample_plain.features_count == 50);
            assert(t.one_sample_plain.compatible(s));
            assert(length(t.one_sample_coded.classes) == 1);
            assert(strcmp(t.one_sample_coded.classes{1},'none'));
            assert(t.one_sample_coded.classes_count == 1);
            assert(tc.check(t.one_sample_coded.samples == A(1,:) - mean(A(1,:))));
            assert(tc.check(t.one_sample_coded.labels_idx == c(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 50);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Substracting DC component from each sample.\n',...
                                                          '  Building dataset.\n'))));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = [mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)';
                 mvnrnd(randi(5) - 3,2,50)'];
            c = ones(4,1);
            s = dataset({'none'},A,c);
            
            t = transforms.dc_offset(s,log);
            s_p = t.code(s,log);
            
            assert(t.one_sample_coded.compatible(s_p));
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(tc.same(s_p.samples,A - repmat(mean(A,2),1,50)));
            assert(tc.check(s_p.labels_idx == c));
            assert(s_p.samples_count == 4);
            assert(s_p.features_count == 50);
            assert(tc.same(mean(s_p.samples,2),zeros(4,1)));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Extracting plain/coded samples:\n',...
                                                          '  Substracting DC component from each sample.\n',...
                                                          '  Building dataset.\n',....
                                                          'Substracting DC component from each sample.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                for ii = 1:4
                    subplot(4,2,(ii - 1)*2 + 1);
                    plot(s.samples(ii,:));
                    axis([1 50 -5 5]);
                    subplot(4,2,(ii - 1)*2 + 2);
                    plot(s_p.samples(ii,:));
                    axis([1 50 -5 5]);
                end
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
