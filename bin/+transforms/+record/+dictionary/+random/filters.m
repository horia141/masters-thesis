classdef filters < transforms.record.dictionary
    methods (Access=public)
        function [obj] = filters(train_sample_plain,word_count,coding_method,coding_params,logger)
            assert(tc.dataset_record(train_sample_plain));
            assert(tc.scalar(word_count));
            assert(tc.natural(word_count));
            assert(word_count > 0);
            assert(tc.scalar(coding_method));
            assert(tc.string(coding_method));
            assert(tc.one_of(coding_method,'Corr','MP','OMP'));
            assert((tc.same(coding_method,'Corr') && tc.empty(coding_params)) || ...
                   (tc.same(coding_method,'MP') && (tc.scalar(coding_params) && ...
                                                    tc.natural(coding_params) && ...
                                                    (coding_params > 0))) || ...
                   (tc.same(coding_method,'OMP') && (tc.scalar(coding_params) && ...
                                                     tc.natural(coding_params) && ...
                                                     (coding_params > 0))));
            assert(tc.scalar(logger));
            assert(tc.logging_logger(logger));
            assert(logger.active);
            
            d = dataset.geometry(train_sample_plain);
            dict = utils.rand_range(-1,1,word_count,d);
            
            obj = obj@transforms.record.dictionary(train_sample_plain,dict,coding_method,coding_params,logger);
        end
    end
    
    methods (Static,Access=public)
        function test(display)
            fprintf('Testing "transforms.record.dictionary.random.filters".\n');
            
            fprintf('  Proper construction.\n');
            
            fprintf('    Correlation.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary.random.filters(s,3,'Corr',[],log);
            
            assert(tc.matrix(t.dict));
            assert(tc.same(size(t.dict),[3 2]));
            assert(tc.number(t.dict));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict(ii,:)),1),1:3));
            assert(tc.matrix(t.dict_transp));
            assert(tc.same(size(t.dict_transp),[2 3]));
            assert(tc.number(t.dict_transp));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict_transp(:,ii)),1),1:3));
            assert(t.word_count == 3);
            assert(tc.same(t.coding_fn,@transforms.record.dictionary.correlation));
            assert(tc.same(t.coding_params,{}));            
            assert(tc.same(t.coding_method,'Corr'));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,3));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Matching Pursuit.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary.random.filters(s,3,'MP',1,log);
            
            assert(tc.matrix(t.dict));
            assert(tc.same(size(t.dict),[3 2]));
            assert(tc.number(t.dict));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict(ii,:)),1),1:3));
            assert(tc.matrix(t.dict_transp));
            assert(tc.same(size(t.dict_transp),[2 3]));
            assert(tc.number(t.dict_transp));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict_transp(:,ii)),1),1:3));
            assert(t.word_count == 3);
            assert(tc.same(t.coding_fn,@transforms.record.dictionary.matching_pursuit));
            assert(tc.same(t.coding_params,{1}));            
            assert(tc.same(t.coding_method,'MP'));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,3));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Orthogonal Matching Pursuit.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary.random.filters(s,3,'OMP',1,log);
            
            assert(tc.matrix(t.dict));
            assert(tc.same(size(t.dict),[3 2]));
            assert(tc.number(t.dict));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict(ii,:)),1),1:3));
            assert(tc.matrix(t.dict_transp));
            assert(tc.same(size(t.dict_transp),[2 3]));
            assert(tc.number(t.dict_transp));
            assert(tc.checkf(@(ii)tc.same(norm(t.dict_transp(:,ii)),1),1:3));
            assert(t.word_count == 3);
            assert(tc.same(t.coding_fn,@transforms.record.dictionary.ortho_matching_pursuit));
            assert(tc.same(t.coding_params,{1}));            
            assert(tc.same(t.coding_method,'OMP'));
            assert(tc.same(t.input_geometry,2));
            assert(tc.same(t.output_geometry,3));

            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "code".\n');
            
            fprintf('    Correlation.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary.random.filters(s,3,'Corr',[],log);
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[3 600]));
            assert(tc.number(s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Matching Pursuit and one kept coefficient.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary.random.filters(s,3,'MP',1,log);
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[3 600]));
            assert(tc.number(s_p));
            assert(tc.same(sum(s_p ~= 0,1),ones(1,600)));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Matching Pursuit and two kept coefficients.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary.random.filters(s,3,'MP',2,log);
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[3 600]));
            assert(tc.number(s_p));
            assert(tc.same(sum(s_p ~= 0,1),2*ones(1,600)));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Orthogonal Matching Pursuit and one kept coefficient.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary.random.filters(s,3,'OMP',1,log);
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[3 600]));
            assert(tc.number(s_p));
            assert(tc.same(sum(s_p ~= 0,1),ones(1,600)));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Orthogonal Matching Pursuit and two kept coefficients.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';
            
            t = transforms.record.dictionary.random.filters(s,3,'OMP',2,log);
            s_p = t.code(s,log);
            
            assert(tc.matrix(s_p));
            assert(tc.same(size(s_p),[3 600]));
            assert(tc.number(s_p));
            assert(tc.same(sum(s_p ~= 0,1),2*ones(1,600)));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,2,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,2,2);
                scatter3(s_p(1,:),s_p(2,:),s_p(3,:),'o','b');
                axis([-7 7 -7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('  Function "decode".\n');
            
            fprintf('    Correlation with three word dictionary.\n');

            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary.random.filters(s,3,'Corr',[],log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
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
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Correlation with two word dictionary.\n');

            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary.random.filters(s,2,'Corr',[],log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                axis([-7 7 -7 7]);
                axis('square');
                title('Original samples.');
                hold off;
                subplot(1,3,2);
                scatter(s_p(1,:),s_p(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Coded samples.');
                subplot(1,3,3);
                scatter(s_r(1,:),s_r(2,:),'o','b');
                axis([-7 7 -7 7]);
                axis('square');
                title('Restored samples.');
                pause(5);
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Matching Pursuit and one kept coefficient.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary.random.filters(s,3,'MP',1,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
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
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Matching Pursuit and two kept coefficients.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary.random.filters(s,3,'MP',2,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
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
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Orthogonal Matching Pursuit and one kept coefficient.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary.random.filters(s,3,'OMP',1,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
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
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
            
            fprintf('    Orthogonal Matching Pursuit and two kept coefficients.\n');
            
            hnd = logging.handlers.testing(logging.level.All);
            log = logging.logger({hnd});
            s = [mvnrnd([0 0],[3 0; 0 0.01],200);mvnrnd([0 0],[0.01 0; 0 3],200);mvnrnd([0 0],[2 1.95; 1.95 2],200)]';

            t = transforms.record.dictionary.random.filters(s,3,'OMP',2,log);
            s_p = t.code(s,log);
            s_r = t.decode(s_p,log);
            
            assert(tc.matrix(s_r));
            assert(tc.same(size(s_r),[2 600]));
            assert(tc.number(s_r));
            assert(tc.same(s_r,t.dict_transp * s_p));
            
            if exist('display','var') && (display == true)
                figure;
                subplot(1,3,1);
                hold on;
                scatter(s(1,:),s(2,:),'o','b');
                line([0;t.dict(1,1)],[0;t.dict(1,2)],'Color','r','LineWidth',3);
                line([0;t.dict(2,1)],[0;t.dict(2,2)],'Color','r','LineWidth',3);
                line([0;t.dict(3,1)],[0;t.dict(3,2)],'Color','r','LineWidth',3);
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
                close(gcf());
            end
            
            log.close();
            hnd.close();
            
            clearvars -except display;
        end
    end
end