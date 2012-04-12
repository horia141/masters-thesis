classdef mean_substract < transforms.reversible
    properties (GetAccess=public,SetAccess=immutable)
        kept_mean;
    end
        
    properties (GetAccess=public,SetAccess=immutable)
        one_sample_plain;
        one_sample_coded;
    end
    
    methods (Access=public)
        function [obj] = mean_substract(train_dataset_plain,logger)
            assert(tc.scalar(train_dataset_plain));
            assert(tc.dataset(train_dataset_plain));
            assert(train_dataset_plain.samples_count >= 1);
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            logger.message('Computing dataset mean.');
            
            kept_mean_t = mean(train_dataset_plain.samples,1);
            
            obj = obj@transforms.reversible(logger);
            obj.kept_mean = kept_mean_t;
            
            logger.beg_node('Extracting plain/coded samples');
            
            obj.one_sample_plain = train_dataset_plain.subsamples(1);
            obj.one_sample_coded = obj.do_code(obj.one_sample_plain,logger);
            
            logger.end_node();
        end
    end
    
    methods (Access=protected)
        function [dataset_coded] = do_code(obj,dataset_plain,logger)
            logger.message('Substracting mean from each sample.');
            
            samples_coded = bsxfun(@minus,dataset_plain.samples,obj.kept_mean);
            
            logger.message('Building dataset.');
            
            dataset_coded = dataset(dataset_plain.classes,samples_coded,dataset_plain.labels_idx);
        end
        
        function [dataset_plain_hat] = do_decode(obj,dataset_coded,logger)
            logger.message('Adding saved mean to each sample.');
            
            samples_plain_hat = bsxfun(@plus,dataset_coded.samples,obj.kept_mean);
            
            logger.message('Building dataset.');
            
            dataset_plain_hat = dataset(dataset_coded.classes,samples_plain_hat,dataset_coded.labels_idx);
        end
    end
        
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.mean_substract".\n');
            
            fprintf('  Proper construction.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = mvnrnd([3 3],[1 0.4; 0.4 0.3],100);
            c = ones(100,1);
            s = dataset({'none'},A,c);
            
            t = transforms.mean_substract(s,log);
            
            assert(tc.same(t.kept_mean,mean(A,1)));
            assert(length(t.one_sample_plain.classes) == 1);
            assert(strcmp(t.one_sample_plain.classes{1},'none'));
            assert(t.one_sample_plain.classes_count == 1);
            assert(tc.check(t.one_sample_plain.samples == A(1,:)));
            assert(tc.check(t.one_sample_plain.labels_idx == c(1)));
            assert(t.one_sample_plain.samples_count == 1);
            assert(t.one_sample_plain.features_count == 2);
            assert(t.one_sample_plain.compatible(s));
            assert(length(t.one_sample_coded.classes) == 1);
            assert(strcmp(t.one_sample_coded.classes{1},'none'));
            assert(t.one_sample_coded.classes_count == 1);
            assert(tc.check(t.one_sample_coded.samples == (A(1,:) - mean(A,1))));
            assert(tc.check(t.one_sample_coded.labels_idx == c(1)));
            assert(t.one_sample_coded.samples_count == 1);
            assert(t.one_sample_coded.features_count == 2);
            assert(t.one_sample_coded.compatible(s));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Substracting mean from each sample.\n',...
                                                          '  Building dataset.\n'))));
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = mvnrnd([3 3],[1 0.4; 0.4 0.4],100);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.mean_substract(s,log);
            s_p = t.code(s,log);
            
            assert(t.one_sample_coded.compatible(s_p));
            assert(length(s_p.classes) == 1);
            assert(strcmp(s_p.classes{1},'none'));
            assert(s_p.classes_count == 1);
            assert(tc.same(s_p.samples,A - repmat(mean(A,1),100,1)));
            assert(tc.check(s_p.labels_idx == c));
            assert(s_p.samples_count == 100);
            assert(s_p.features_count == 2);
            assert(tc.same(mean(s_p.samples,1),[0 0]));
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Substracting mean from each sample.\n',...
                                                          '  Building dataset.\n',...
                                                          'Substracting mean from each sample.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,2,1);
                scatter(s.samples(:,1),s.samples(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,2,2);
                scatter(s_p.samples(:,1),s_p.samples(:,2),'x');
                axis([-4 6 -4 6]);
                title('Mean substracted samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            A = mvnrnd([3 3],[1 0.4; 0.4 0.4],100);
            c = ones(100,1);            
            s = dataset({'none'},A,c);
            
            t = transforms.mean_substract(s,log);
            s_p = t.code(s,log);            
            s_r = t.decode(s_p,log);
            
            assert(t.one_sample_coded.compatible(s_p));
            assert(length(s_r.classes) == 1);
            assert(strcmp(s_r.classes{1},'none'));
            assert(s_r.classes_count == 1);
            assert(tc.same(s_r.samples,A));
            assert(tc.check(s_r.labels_idx == c));
            assert(s_r.samples_count == 100);
            assert(s_r.features_count == 2);
            
            assert(tc.same(hnd.logged_data,sprintf(strcat('Computing dataset mean.\n',...
                                                          'Extracting plain/coded samples:\n',...
                                                          '  Substracting mean from each sample.\n',...
                                                          '  Building dataset.\n',...
                                                          'Substracting mean from each sample.\n',...
                                                          'Building dataset.\n',...
                                                          'Adding saved mean to each sample.\n',...
                                                          'Building dataset.\n'))));
            
            if exist('display','var') && (display == true)
                figure();
                subplot(1,3,1);
                scatter(s.samples(:,1),s.samples(:,2),'o');
                axis([-4 6 -4 6]);
                title('Original samples.');
                subplot(1,3,2);
                scatter(s_p.samples(:,1),s_p.samples(:,2),'x');
                axis([-4 6 -4 6]);
                title('Mean substracted samples.');
                subplot(1,3,3);
                scatter(s_r.samples(:,1),s_r.samples(:,2),'.');
                axis([-4 6 -4 6]);
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end
